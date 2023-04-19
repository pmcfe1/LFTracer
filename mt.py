import inspect
from collections import Counter
import copy
import sys
from typing import Any, Optional, Callable, Dict, List, Type, TextIO, cast
from types import FrameType, TracebackType
import traceback
import numpy


class LFTracer():
    def __init__(self, target_func: str = [], list_func: bool = False, file: TextIO = sys.stdout) -> None:
        self.target_func = target_func
        self.lft_counter = Counter()
        self.last_vars: Dict[str, Any] = {}
        self.list_func = list_func

        self.original_trace_function: Optional[Callable] = None
        self.file = file

    def changed_statement(self, statements: Dict[str, Any], frame: FrameType) -> Dict[str, Any]:
        #if frame.f_code.co_name == self.target_func[0]:
        for target_fun, target_value in statements.items():
            should_change = False
            if target_fun in self.last_vars and (type(self.last_vars[target_fun]) == numpy.ndarray or type(target_value) == numpy.ndarray):
                if (self.last_vars[target_fun]!=target_value).all():
                    should_change = True
            elif (target_fun not in self.last_vars or
                    self.last_vars[target_fun] != target_value):
                should_change = True

            if should_change:
                if target_fun in self.lft_counter:
                    self.lft_counter[target_fun] += 1
                else:
                    self.lft_counter[target_fun] = 1

        self.last_vars = copy.deepcopy(statements)
        return self.last_vars

    def traceit(self, frame: FrameType, event: str, arg: Any) -> None:
        if frame.f_code.co_name == self.target_func[0] or frame.f_code.co_name == self.target_func[1]:
            if event == 'line':
                module = inspect.getmodule(frame.f_code)
                if module == 'if' or 'for' or 'while' or 'return':
                    self.changed_statement(frame.f_locals, frame)

        if event == 'return' and (frame.f_code.co_name == self.target_func[0] or frame.f_code.co_name == self.target_func[1]):
            self.last_vars = {}  # Delete 'last' variables if the target_func returns

    def _traceit(self, frame: FrameType, event: str, arg: Any) -> Optional[Callable]:
        if self.our_frame(frame):
            # Do not trace our own methods
            pass
        else:
            self.traceit(frame, event, arg)
        return self._traceit

    def __enter__(self) -> Any:
        self.original_trace_function = sys.gettrace()
        sys.settrace(self._traceit)

        return self

    def __exit__(self, exc_tp: Type, exc_value: BaseException, exc_traceback: TracebackType) -> Optional[bool]:
        sys.settrace(self.original_trace_function)

        if self.is_internal_error(exc_tp, exc_value, exc_traceback):
            return False  # internal error
        else:
            return None  # all ok

    def is_internal_error(self, exc_tp: Type,
                          exc_value: BaseException,
                          exc_traceback: TracebackType) -> bool:
        if not exc_tp:
            return False

        for frame, lineno in traceback.walk_tb(exc_traceback):
            if self.our_frame(frame):
                return True

        return False

    def getLFMap(self):
        return self.lft_counter

    def our_frame(self, frame: FrameType) -> bool:
        return isinstance(frame.f_locals.get('self'), self.__class__)


    # R1. Count the number of visiting for each statement in the target functions.
    # R2. There could be multiple target functions.
