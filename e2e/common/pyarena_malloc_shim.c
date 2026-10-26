/* LD_PRELOAD shim: CPython's arena allocator (pymalloc arenas and frame data-stack chunks) uses malloc/free instead of
   mmap/munmap, so arenas are reused instead of unmapped. No-op in non-Python processes. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
typedef struct { void *ctx; void *(*alloc)(void *ctx, size_t size); void (*free)(void *ctx, void *ptr, size_t size); } PyObjectArenaAllocator;
static void *shim_alloc(void *ctx, size_t size) { (void)ctx; return malloc(size); }
static void shim_free(void *ctx, void *ptr, size_t size) { (void)ctx; (void)size; free(ptr); }
__attribute__((constructor)) static void pyarena_shim_init(void) {
  void (*setter)(PyObjectArenaAllocator *) = (void (*)(PyObjectArenaAllocator *))dlsym(RTLD_DEFAULT, "PyObject_SetArenaAllocator");
  if (!setter) return;
  PyObjectArenaAllocator a = { NULL, shim_alloc, shim_free };
  setter(&a);
  if (getenv("PYARENA_SHIM_VERBOSE")) fprintf(stderr, "[pyarena_malloc_shim] active in pid %d: CPython arena allocator -> malloc/free\n", (int)getpid());
}
