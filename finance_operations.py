import numpy as np
import pandas as pd

values_to_change = ["open", "close", "high", "low"]

def get_adjusted_values(values, basepoint=0):
    changes_on_values = [(x["split"], x["dividend"]) for x in values]
    if len([value for value in changes_on_values if value[0] != 1 or value[1] != 0]) == 0:
        return values
    else:
        new_values_next = []
        new_values_prev = []
        operations = []
        for i in range(basepoint+1, len(values)):
            if changes_on_values[i][1] > 0:
                operations.append(("+", changes_on_values[i][1]))
            if changes_on_values[i][0] != 1:
                operations.append(("*", changes_on_values[i][0]))

            current_value = values[i].copy()
            for operation in operations:
                if operation[0] == "+":
                    for dict_name in values_to_change:
                        current_value[dict_name] += operation[1]
                else:
                    for dict_name in values_to_change:
                        current_value[dict_name] *= operation[1]
            new_values_next.append(current_value)
        operations = []

        for i in range(0, basepoint+1):
            index = basepoint-i

            current_value = values[index].copy()
            for operation in operations:
                if operation[0] == "-":
                    for dict_name in values_to_change:
                        current_value[dict_name] -= operation[1]
                else:
                    for dict_name in values_to_change:
                        current_value[dict_name] /= operation[1]
            new_values_prev.append(current_value)
            if changes_on_values[index][0] != 1:
                operations.append(("/", changes_on_values[index][0]))
            if changes_on_values[index][1] > 0:
                operations.append(("-", changes_on_values[index][1]))
        return list(reversed(new_values_prev)) + new_values_next