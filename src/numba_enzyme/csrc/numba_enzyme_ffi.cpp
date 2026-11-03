// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// In-process LLVM IR extraction, linking, and Enzyme differentiation entry
// point for numba-enzyme. Exports a tiny C ABI (`numba_enzyme_compile_ir` and
// `numba_enzyme_free_string`) with all internal LLVM and Enzyme symbols hidden
// so loading into a Python process alongside `llvmlite` never causes symbol
// collisions on Linux or macOS.

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/ExtractGV.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/IPO/StripSymbols.h"

extern "C" void registerEnzymeAndPassPipeline(llvm::PassBuilder &PB,
                                              bool augment);

namespace {

char *dup_cstr(llvm::StringRef S) {
  char *Buf = static_cast<char *>(std::malloc(S.size() + 1));
  if (!Buf) {
    return nullptr;
  }
  if (!S.empty()) {
    std::memcpy(Buf, S.data(), S.size());
  }
  Buf[S.size()] = '\0';
  return Buf;
}

void collect_reachable_globals(
    llvm::Value *V,
    llvm::SmallPtrSetImpl<const llvm::Value *> &Visited,
    llvm::SetVector<llvm::GlobalValue *> &GVs) {
  if (!V || !Visited.insert(V).second) {
    return;
  }
  if (auto *F = llvm::dyn_cast<llvm::Function>(V)) {
    if (!F->isDeclaration()) {
      GVs.insert(F);
      for (llvm::BasicBlock &BB : *F) {
        for (llvm::Instruction &I : BB) {
          for (llvm::Value *Op : I.operands()) {
            collect_reachable_globals(Op, Visited, GVs);
          }
        }
      }
    }
    return;
  }
  if (auto *GV = llvm::dyn_cast<llvm::GlobalVariable>(V)) {
    if (!GV->isDeclaration()) {
      GVs.insert(GV);
      if (GV->hasInitializer()) {
        collect_reachable_globals(GV->getInitializer(), Visited, GVs);
      }
    }
    return;
  }
  if (auto *GA = llvm::dyn_cast<llvm::GlobalAlias>(V)) {
    if (!GA->isDeclaration()) {
      GVs.insert(GA);
      collect_reachable_globals(GA->getAliasee(), Visited, GVs);
    }
    return;
  }
  if (auto *U = llvm::dyn_cast<llvm::User>(V)) {
    for (llvm::Value *Op : U->operands()) {
      collect_reachable_globals(Op, Visited, GVs);
    }
  }
}

void diagnostic_callback(const llvm::DiagnosticInfo *DI, void *Ctx) {
  if (!DI || !Ctx) {
    return;
  }
  if (DI->getSeverity() == llvm::DS_Error) {
    auto *ErrStr = static_cast<std::string *>(Ctx);
    llvm::raw_string_ostream OS(*ErrStr);
    llvm::DiagnosticPrinterRawOStream DP(OS);
    DI->print(DP);
    OS << "\n";
  }
}

std::string format_sm_diagnostic(const llvm::SMDiagnostic &Diag,
                                 const char *BufferName) {
  std::string Msg;
  llvm::raw_string_ostream OS(Msg);
  Diag.print(BufferName, OS, /*ShowColors=*/false);
  return Msg;
}

std::string print_module_to_string(const llvm::Module &M) {
  std::string Out;
  llvm::raw_string_ostream OS(Out);
  M.print(OS, nullptr);
  return Out;
}

}  // namespace

#if defined(_WIN32)
#define NUMBA_ENZYME_EXPORT __declspec(dllexport)
#else
#define NUMBA_ENZYME_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

NUMBA_ENZYME_EXPORT int numba_enzyme_compile_ir(
    const char *kernel_ir,
    const char *entry_symbol,
    const char *driver_ir,
    char **combined_ir_out,
    char **enzyme_ir_out,
    char **error_out) {
  if (combined_ir_out) {
    *combined_ir_out = nullptr;
  }
  if (enzyme_ir_out) {
    *enzyme_ir_out = nullptr;
  }
  if (error_out) {
    *error_out = nullptr;
  }

  if (!kernel_ir || !entry_symbol || !driver_ir || !enzyme_ir_out) {
    if (error_out) {
      *error_out =
          dup_cstr("numba_enzyme_compile_ir received a null argument");
    }
    return 1;
  }

  llvm::LLVMContext Context;
  std::string DiagErrors;
  Context.setDiagnosticHandlerCallBack(diagnostic_callback, &DiagErrors);

  llvm::SMDiagnostic ParseErr;
  std::unique_ptr<llvm::Module> KernelMod = llvm::parseIR(
      llvm::MemoryBufferRef(kernel_ir, "kernel_ir"), ParseErr, Context);
  if (!KernelMod) {
    if (error_out) {
      *error_out = dup_cstr(format_sm_diagnostic(ParseErr, "kernel_ir"));
    }
    return 1;
  }

  llvm::Function *EntryFn = KernelMod->getFunction(entry_symbol);
  if (!EntryFn || EntryFn->isDeclaration()) {
    if (error_out) {
      std::string Msg =
          std::string("entry symbol not found or has no body: ") + entry_symbol;
      *error_out = dup_cstr(Msg);
    }
    return 1;
  }

  // Extract `entry_symbol` and all transitively reachable defined functions
  // and globals (mirroring `llvm-extract --func=<entry> --recursive
  // --keep-const-init`), discarding Numba's Python/cfunc wrapper and its
  // Python C-API declarations.
  llvm::SmallPtrSet<const llvm::Value *, 32> Visited;
  llvm::SetVector<llvm::GlobalValue *> ExtractedGVs;
  collect_reachable_globals(EntryFn, Visited, ExtractedGVs);
  std::vector<llvm::GlobalValue *> GVVec(ExtractedGVs.begin(),
                                         ExtractedGVs.end());

  {
    llvm::ModuleAnalysisManager ExtractMAM;
    llvm::PassBuilder ExtractPB;
    ExtractPB.registerModuleAnalyses(ExtractMAM);

    llvm::ModulePassManager ExtractMPM;
    ExtractMPM.addPass(llvm::ExtractGVPass(GVVec, /*deleteS=*/false,
                                           /*keepConstInit=*/true));
    ExtractMPM.addPass(llvm::GlobalDCEPass());
    ExtractMPM.addPass(llvm::StripDeadDebugInfoPass());
    ExtractMPM.addPass(llvm::StripDeadPrototypesPass());
    ExtractMPM.run(*KernelMod, ExtractMAM);
  }

  std::unique_ptr<llvm::Module> DriverMod = llvm::parseIR(
      llvm::MemoryBufferRef(driver_ir, "driver_ir"), ParseErr, Context);
  if (!DriverMod) {
    if (error_out) {
      *error_out = dup_cstr(format_sm_diagnostic(ParseErr, "driver_ir"));
    }
    return 1;
  }

  DiagErrors.clear();
  if (llvm::Linker::linkModules(*KernelMod, std::move(DriverMod))) {
    if (error_out) {
      *error_out = dup_cstr(
          DiagErrors.empty() ? "llvm::Linker::linkModules failed" : DiagErrors);
    }
    return 1;
  }

  if (combined_ir_out) {
    *combined_ir_out = dup_cstr(print_module_to_string(*KernelMod));
  }

  // Run the Enzyme differentiation pass.
  {
    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;

    llvm::PassBuilder PB;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    registerEnzymeAndPassPipeline(PB, /*augment=*/false);

    llvm::ModulePassManager MPM;
    if (llvm::Error E = PB.parsePassPipeline(MPM, "enzyme")) {
      std::string Msg = llvm::toString(std::move(E));
      if (error_out) {
        *error_out = dup_cstr(Msg);
      }
      return 1;
    }

    DiagErrors.clear();
    MPM.run(*KernelMod, MAM);
    if (!DiagErrors.empty()) {
      if (error_out) {
        *error_out = dup_cstr(DiagErrors);
      }
      return 1;
    }
  }

  std::string VerifyMsg;
  llvm::raw_string_ostream VerifyOS(VerifyMsg);
  if (llvm::verifyModule(*KernelMod, &VerifyOS)) {
    if (error_out) {
      *error_out = dup_cstr("Enzyme output failed LLVM IR verification:\n" +
                            VerifyOS.str());
    }
    return 1;
  }

  *enzyme_ir_out = dup_cstr(print_module_to_string(*KernelMod));
  if (!*enzyme_ir_out) {
    if (error_out) {
      *error_out = dup_cstr("out of memory serializing Enzyme IR");
    }
    return 1;
  }
  return 0;
}

NUMBA_ENZYME_EXPORT void numba_enzyme_free_string(char *s) { std::free(s); }

}  // extern "C"
