class AgentInfo:
    """Dataclass that stores the column associated with each agent attribute"""

    def __init__(self, columns):
        self._columns = columns  # Store original column names

        for index, name in enumerate(columns):
            setattr(self, name, index)

    def __getitem__(self, key):
        """Allow dict-like access."""
        if isinstance(key, str):
            return getattr(self, key, None)
        elif isinstance(key, int) and 0 <= key < len(self._columns):
            return self._columns[key]
        raise KeyError(f"Invalid key: {key}")

    def add(self, name, index):
        """Add a new column."""
        if name in self._columns:
            raise ValueError(f"Column {name} already exists.")

        setattr(self, name, index)
        self._columns.append(name)

    def __repr__(self):
        """Readable representation."""
        return f"ColumnIndexer({self._columns})"

    def keys(self):
        """Return column names."""
        return self._columns

    def values(self):
        """Return column indices."""
        return list(range(len(self._columns)))

    def items(self):
        """Return (column_name, index) pairs."""
        return zip(self._columns, self.values())

    def __iter__(self):
        """Iterate over column names."""
        return iter(self._columns)
