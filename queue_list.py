from typing import List, TypeVar, Optional


class QueueList(list):
    """This is custom queue list container to manage order of frames to analyse."""

    _T = TypeVar("_T")

    def __init__(self, max_size: int = 15):
        """
        The QueueList class constructor.

        :param max_size: (int) max size of queue list.
        """
        list.__init__(self)

        self._list_object = []
        self._max_size = max_size

    @property
    def max_size(self) -> int:
        """
        There is property to get maximum size of current container object.

        :return: (int) maximum size of queue.
        """
        return self._max_size

    @max_size.setter
    def max_size(self, value: int):
        """
        There is setter for `self._max_size` attribute.

        :param value: (int) new size.
        """
        self.clear()
        self._max_size = value

    @property
    def to_list(self) -> List[_T]:
        """
        There is property to get list object from current queue.

        :return: (List[_T]) new list object from current queue.
        """
        return self._list_object

    def __len__(self):
        """Returns the length of queue list."""
        return self._list_object.__len__()

    def __repr__(self):
        """Returns string value describes current class."""
        return f'<QueueList: {self._list_object.__len__()}>'

    def __getitem__(self, item):
        """
        Returns the list object item by item.

        :param item: (_T) item to get from list.
        """
        return self._list_object[item]

    def get_last(self) -> _T:
        """Returns the last element of queue list."""
        return self._list_object[-1:][0]

    def pop_first(self) -> Optional[_T]:
        """Returns the first element of queue list."""
        return self._list_object.pop(0) \
            if len(self._list_object) > 0 \
            else None

    def pop_last(self) -> _T:
        """Returns the last element of queue list."""
        return self._list_object.pop(len(self) - 1)

    def pop_pre_last(self) -> _T:
        """Returns the pre-last element of queue list."""
        return self._list_object.pop(len(self) - 2)

    def append(self, _object: _T) -> None:
        """
        Pushes passed object to queue list.

        :param _object: (_T) object to append to list.
        """
        if len(self._list_object) >= self._max_size:
            _offset = self._max_size - 1
            self._list_object = self._list_object[-_offset:]

        self._list_object.append(_object)
