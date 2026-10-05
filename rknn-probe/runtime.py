"""Fixed-shape test binding to the explicitly selected SDK 2.3.2 C runtime."""
import ctypes as C
from pathlib import Path

import numpy as np

LIB = str(Path(__file__).resolve().parents[1] / "rknn-2.3.2/lib/librknnrt.so")


class Input(C.Structure):
    _fields_ = [("index", C.c_uint32), ("buf", C.c_void_p), ("size", C.c_uint32),
                ("pass_through", C.c_uint8), ("type", C.c_int), ("fmt", C.c_int)]


class Output(C.Structure):
    _fields_ = [("want_float", C.c_uint8), ("is_prealloc", C.c_uint8),
                ("index", C.c_uint32), ("buf", C.c_void_p), ("size", C.c_uint32)]


class Version(C.Structure):
    _fields_ = [("api", C.c_char * 256), ("driver", C.c_char * 256)]


class Detail(C.Structure):
    _fields_ = [("data", C.c_void_p), ("length", C.c_uint64)]


class TensorAttr(C.Structure):
    _fields_ = [("index", C.c_uint32), ("n_dims", C.c_uint32), ("dims", C.c_uint32 * 16),
                ("name", C.c_char * 256), ("n_elems", C.c_uint32), ("size", C.c_uint32),
                ("fmt", C.c_int), ("type", C.c_int), ("qnt_type", C.c_int), ("fl", C.c_int8),
                ("zp", C.c_int32), ("scale", C.c_float), ("w_stride", C.c_uint32),
                ("size_with_stride", C.c_uint32), ("pass_through", C.c_uint8), ("h_stride", C.c_uint32)]


class MemorySize(C.Structure):
    _fields_ = [("weight_bytes", C.c_uint32), ("internal_bytes", C.c_uint32),
                ("dma_allocated_bytes", C.c_uint64), ("sram_bytes", C.c_uint32),
                ("free_sram_bytes", C.c_uint32), ("reserved", C.c_uint32 * 10)]


def check(ret):
    assert ret == 0, ret


class Runtime:
    def __init__(self, model, shapes, flags=0):
        assert C.sizeof(Input) == 32 and C.sizeof(Output) == 24
        self.shapes = shapes
        self.model_path = str(Path(model).resolve())
        self.lib = C.CDLL(LIB)
        signatures = {
            "rknn_init": [C.POINTER(C.c_uint64), C.c_void_p, C.c_uint32, C.c_uint32, C.c_void_p],
            "rknn_inputs_set": [C.c_uint64, C.c_uint32, C.POINTER(Input)],
            "rknn_run": [C.c_uint64, C.c_void_p],
            "rknn_outputs_get": [C.c_uint64, C.c_uint32, C.POINTER(Output), C.c_void_p],
            "rknn_outputs_release": [C.c_uint64, C.c_uint32, C.POINTER(Output)],
            "rknn_query": [C.c_uint64, C.c_int, C.c_void_p, C.c_uint32],
            "rknn_set_core_mask": [C.c_uint64, C.c_int],
            "rknn_destroy": [C.c_uint64],
        }
        for fn, args in signatures.items():
            getattr(self.lib, fn).argtypes, getattr(self.lib, fn).restype = args, C.c_int
        self.ctx = C.c_uint64()
        filename = C.create_string_buffer(str(Path(model).resolve()).encode())
        check(self.lib.rknn_init(C.byref(self.ctx), filename, 0, flags, None))
        try:
            check(self.lib.rknn_set_core_mask(self.ctx, 0xffff))
            version = Version()
            check(self.lib.rknn_query(self.ctx, 5, C.byref(version), C.sizeof(version)))
            self.version, self.driver = version.api.decode(), version.driver.decode()
            assert self.version.startswith("2.3.2"), self.version
            self.mapped = {line.split(maxsplit=5)[5].strip()
                           for line in Path("/proc/self/maps").read_text().splitlines()
                           if "librknnrt.so" in line}
            assert self.mapped == {LIB}, self.mapped
            counts = (C.c_uint32 * 2)()
            check(self.lib.rknn_query(self.ctx, 0, counts, C.sizeof(counts)))
            assert counts[1] == len(shapes)
            self.input_attrs = []
            for i in range(counts[0]):
                attr = TensorAttr(index=i)
                check(self.lib.rknn_query(self.ctx, 1, C.byref(attr), C.sizeof(attr)))
                self.input_attrs.append({"name": attr.name.decode(), "shape": list(attr.dims[:attr.n_dims]),
                                         "type": attr.type, "format": attr.fmt,
                                         "quantization_type": attr.qnt_type, "scale": attr.scale, "zero_point": attr.zp})
            self.output_attrs = []
            for i in range(counts[1]):
                attr = TensorAttr(index=i)
                check(self.lib.rknn_query(self.ctx, 2, C.byref(attr), C.sizeof(attr)))
                self.output_attrs.append({"name": attr.name.decode(), "shape": list(attr.dims[:attr.n_dims]),
                    "type": attr.type, "format": attr.fmt, "quantization_type": attr.qnt_type,
                    "scale": attr.scale, "zero_point": attr.zp})
        except Exception:
            self.close()
            raise

    def run(self, values):
        # Runtime normalization accepts NHWC source for 4D inputs; keep semantic ONNX order
        # by transposing NCHW arrays before that conversion. Non-image ranks stay contiguous.
        arrays = [np.ascontiguousarray(value.transpose(0, 2, 3, 1) if value.ndim == 4 else value)
                  for value in values]
        assert len(arrays) == len(self.input_attrs)
        inputs = (Input * len(arrays))()
        for i, array in enumerate(arrays):
            assert array.dtype in (np.float32, np.int64), array.dtype
            inputs[i] = Input(i, array.ctypes.data, array.nbytes, 0,
                              0 if array.dtype == np.float32 else 8, 1 if array.ndim == 4 else 0)
        check(self.lib.rknn_inputs_set(self.ctx, len(inputs), inputs))
        check(self.lib.rknn_run(self.ctx, None))
        outputs = (Output * len(self.shapes))()
        for i in range(len(outputs)):
            outputs[i].want_float, outputs[i].index = 1, i
        check(self.lib.rknn_outputs_get(self.ctx, len(outputs), outputs, None))
        try:
            result = []
            for output, shape in zip(outputs, self.shapes):
                count = int(np.prod(shape))
                assert output.size == count * 4, (output.size, shape)
                array = np.ctypeslib.as_array(C.cast(output.buf, C.POINTER(C.c_float)), shape=(count,))
                result.append(array.reshape(shape).copy())
            return result
        finally:
            check(self.lib.rknn_outputs_release(self.ctx, len(outputs), outputs))

    def profile(self):
        detail = Detail()
        check(self.lib.rknn_query(self.ctx, 3, C.byref(detail), C.sizeof(detail)))
        return C.string_at(detail.data, detail.length).decode(errors="replace").rstrip("\x00")

    def memory(self):
        value = MemorySize()
        assert C.sizeof(value) == 64
        check(self.lib.rknn_query(self.ctx, 6, C.byref(value), C.sizeof(value)))
        return {name: getattr(value, name) for name, _ in value._fields_ if name != "reserved"}

    def close(self):
        if self.ctx.value:
            check(self.lib.rknn_destroy(self.ctx))
            self.ctx.value = 0


if __name__ == "__main__":
    import json
    import sys
    runtime = Runtime(sys.argv[1], [(1, 6), (1, 768)])
    try:
        print(json.dumps({"sdk": runtime.version, "memory": runtime.memory(),
                          "inputs": runtime.input_attrs, "outputs": runtime.output_attrs}, indent=2), flush=True)
    finally:
        runtime.close()
