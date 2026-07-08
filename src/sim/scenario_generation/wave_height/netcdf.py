from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NC_DIMENSION = 10
NC_VARIABLE = 11
NC_ATTRIBUTE = 12
NC_ABSENT = 0

TYPE_NAMES = {
    1: "byte",
    2: "char",
    3: "short",
    4: "int",
    5: "float",
    6: "double",
}
TYPE_SIZES = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 4,
    6: 8,
}
TYPE_FORMATS = {
    1: "b",
    3: "h",
    4: "i",
    5: "f",
    6: "d",
}


@dataclass(frozen=True)
class NetCDFVariable:
    name: str
    dimensions: tuple[str, ...]
    type_id: int
    type_name: str
    vsize: int
    begin: int
    attributes: dict[str, Any]

    @property
    def is_record_variable(self) -> bool:
        return bool(self.dimensions) and self.dimensions[0] == "time"


class _HeaderReader:
    def __init__(self, handle) -> None:
        self.handle = handle

    def read(self, size: int) -> bytes:
        data = self.handle.read(size)
        if len(data) != size:
            raise EOFError(f"Expected {size} bytes, got {len(data)}.")
        return data

    def i32(self) -> int:
        return struct.unpack(">i", self.read(4))[0]

    def u32(self) -> int:
        return struct.unpack(">I", self.read(4))[0]

    def i64(self) -> int:
        return struct.unpack(">q", self.read(8))[0]

    def string(self) -> str:
        size = self.i32()
        data = self.read(size)
        padding = (-size) % 4
        if padding:
            self.read(padding)
        return data.decode("utf-8", errors="replace")

    def values(self, type_id: int, count: int) -> Any:
        size = TYPE_SIZES[type_id] * count
        data = self.read(size)
        padding = (-size) % 4
        if padding:
            self.read(padding)
        if type_id == 2:
            return data.decode("utf-8", errors="replace").rstrip("\x00")
        if count == 0:
            return ()
        if count > 64:
            return data
        values = struct.unpack(">" + TYPE_FORMATS[type_id] * count, data)
        return values[0] if count == 1 else values


class ClassicNetCDF:
    """Minimal reader for classic NetCDF CDF-1/CDF-2 weather files.

    The reader intentionally supports only the operations needed here: inspect
    dimensions/attributes, read 2-D coordinate grids, and read one record slice
    of a record variable such as ``significant_wave_height``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.version: int = 0
        self.num_records: int = 0
        self.dimensions: dict[str, int] = {}
        self.unlimited_dimension: str | None = None
        self.global_attributes: dict[str, Any] = {}
        self.variables: dict[str, NetCDFVariable] = {}
        self.record_size: int = 0
        self._parse_header()

    def variable(self, name: str) -> NetCDFVariable:
        try:
            return self.variables[name]
        except KeyError as exc:
            raise KeyError(f"Variable {name!r} not found in {self.path}.") from exc

    def read_grid(self, variable_name: str) -> list[float]:
        variable = self.variable(variable_name)
        if variable.is_record_variable:
            raise ValueError(f"{variable_name!r} is a record variable; use read_record_grid().")
        count = self._variable_value_count(variable)
        return self._read_values(variable, variable.begin, count)

    def read_record_grid(self, variable_name: str, record_index: int) -> list[float]:
        variable = self.variable(variable_name)
        if not variable.is_record_variable:
            raise ValueError(f"{variable_name!r} is not a record variable.")
        if not 0 <= record_index < self.num_records:
            raise IndexError(f"Record index {record_index} outside 0..{self.num_records - 1}.")
        count = self._variable_value_count(variable, skip_unlimited=True)
        offset = variable.begin + record_index * self.record_size
        return self._read_values(variable, offset, count)

    def read_record_values(
        self,
        variable_name: str,
        record_index: int,
        flat_indices: list[int] | tuple[int, ...],
    ) -> list[float]:
        variable = self.variable(variable_name)
        if not variable.is_record_variable:
            raise ValueError(f"{variable_name!r} is not a record variable.")
        if variable.type_id not in TYPE_FORMATS:
            raise TypeError(f"Unsupported numeric type for {variable_name!r}: {variable.type_name}.")
        item_size = TYPE_SIZES[variable.type_id]
        fmt = ">" + TYPE_FORMATS[variable.type_id]
        base = variable.begin + record_index * self.record_size
        values: list[float] = []
        with self.path.open("rb") as handle:
            for index in flat_indices:
                handle.seek(base + index * item_size)
                values.append(float(struct.unpack(fmt, handle.read(item_size))[0]))
        return values

    def fill_value(self, variable_name: str) -> float | None:
        attrs = self.variable(variable_name).attributes
        value = attrs.get("_FillValue", attrs.get("missing_value"))
        return float(value) if isinstance(value, (int, float)) else None

    def _parse_header(self) -> None:
        with self.path.open("rb") as handle:
            reader = _HeaderReader(handle)
            magic = reader.read(4)
            if magic[:3] != b"CDF" or magic[3] not in (1, 2):
                raise ValueError(f"{self.path} is not a classic NetCDF CDF-1/CDF-2 file.")
            self.version = magic[3]
            self.num_records = reader.u32()
            self.dimensions = self._read_dimensions(reader)
            self.global_attributes = self._read_attributes(reader)
            self.variables = self._read_variables(reader)
            self.record_size = sum(v.vsize for v in self.variables.values() if v.is_record_variable)

    def _read_dimensions(self, reader: _HeaderReader) -> dict[str, int]:
        tag = reader.i32()
        if tag == NC_ABSENT:
            reader.i32()
            return {}
        if tag != NC_DIMENSION:
            raise ValueError(f"Unexpected NetCDF dimension tag: {tag}.")
        dimensions: dict[str, int] = {}
        for _ in range(reader.i32()):
            name = reader.string()
            size = reader.u32()
            if size == 0:
                self.unlimited_dimension = name
                dimensions[name] = self.num_records
            else:
                dimensions[name] = size
        return dimensions

    def _read_attributes(self, reader: _HeaderReader) -> dict[str, Any]:
        tag = reader.i32()
        if tag == NC_ABSENT:
            reader.i32()
            return {}
        if tag != NC_ATTRIBUTE:
            raise ValueError(f"Unexpected NetCDF attribute tag: {tag}.")
        attributes: dict[str, Any] = {}
        for _ in range(reader.i32()):
            name = reader.string()
            type_id = reader.i32()
            count = reader.i32()
            attributes[name] = reader.values(type_id, count)
        return attributes

    def _read_variables(self, reader: _HeaderReader) -> dict[str, NetCDFVariable]:
        tag = reader.i32()
        if tag == NC_ABSENT:
            reader.i32()
            return {}
        if tag != NC_VARIABLE:
            raise ValueError(f"Unexpected NetCDF variable tag: {tag}.")
        names = list(self.dimensions)
        variables: dict[str, NetCDFVariable] = {}
        for _ in range(reader.i32()):
            name = reader.string()
            dim_count = reader.i32()
            dimension_ids = [reader.i32() for _ in range(dim_count)]
            attrs = self._read_attributes(reader)
            type_id = reader.i32()
            vsize = reader.u32()
            begin = reader.i32() if self.version == 1 else reader.i64()
            dimensions = tuple(names[index] for index in dimension_ids)
            variables[name] = NetCDFVariable(
                name=name,
                dimensions=dimensions,
                type_id=type_id,
                type_name=TYPE_NAMES.get(type_id, str(type_id)),
                vsize=vsize,
                begin=begin,
                attributes=attrs,
            )
        return variables

    def _variable_value_count(self, variable: NetCDFVariable, *, skip_unlimited: bool = False) -> int:
        count = 1
        dimensions = variable.dimensions[1:] if skip_unlimited else variable.dimensions
        for dimension in dimensions:
            count *= self.dimensions[dimension]
        return count

    def _read_values(self, variable: NetCDFVariable, offset: int, count: int) -> list[float]:
        if variable.type_id not in TYPE_FORMATS:
            raise TypeError(f"Unsupported numeric type for {variable.name!r}: {variable.type_name}.")
        size = TYPE_SIZES[variable.type_id] * count
        with self.path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(size)
        if len(data) != size:
            raise EOFError(f"Expected {size} bytes for {variable.name!r}, got {len(data)}.")
        values = struct.unpack(">" + TYPE_FORMATS[variable.type_id] * count, data)
        return [float(value) for value in values]
