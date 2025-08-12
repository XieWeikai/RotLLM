import ppl
import piqa
import WinoGrande
import HellaSwag
import ARCe
import ARCc
import LAMBADA
import traceback

def save_result(task_name, result, filename="test_results_phi.txt"):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(f"{task_name}: {result}\n")

def save_error(task_name, error_msg, filename="test_results_phi.txt"):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(f"{task_name} ERROR: {error_msg}\n")

def main():
    # 清空结果文件（可选）
    with open("test_results_phi.txt", 'w', encoding='utf-8') as f:
        f.write("=== Test Results ===\n")

    modules = [
        ('ppl', ppl),
        ('piqa', piqa),
        ('WinoGrande', WinoGrande),
        ('HellaSwag', HellaSwag),
        ('ARCe', ARCe),
        ('ARCc', ARCc),
        ('LAMBADA', LAMBADA)
    ]

    results = {}

    for task_name, module in modules:
        print(f"Running {task_name}.main()")
        try:
            result = module.main()
            save_result(task_name, result)
            results[task_name] = result
        except Exception as e:
            err_trace = traceback.format_exc()
            save_error(task_name, err_trace)
            print(f"{task_name} raised an error, saved error info to file.")

    # 终端打印汇总
    print("\n===== Summary of successful results =====")
    for task, result in results.items():
        print(f"{task}: {result}")

if __name__ == "__main__":
    main()
