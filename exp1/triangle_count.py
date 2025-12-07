import time
import random


def triangle_number_method1(nums):
    """
    方法一：枚举最长边 + 相向双指针
    
    时间复杂度：O(n^2)
    空间复杂度：O(1)
    
    算法思路：
    1. 先对数组从小到大排序
    2. 外层循环枚举最长边 c=nums[k]
    3. 内层循环用相向双指针枚举 a=nums[i] 和 b=nums[j]
    4. 初始化左右指针 i=0, j=k-1
    5. 如果 nums[i]+nums[j]>c，找到 j-i 个合法方案，j减一
    6. 如果 nums[i]+nums[j]<=c，i加一
    7. 重复直到 i>=j
    
    参数：
        nums: 包含非负整数的数组
    
    返回：
        能组成三角形的三元组个数
    """
    if len(nums) < 3:
        return 0
    
    # 先对数组从小到大排序
    nums.sort()
    count = 0
    n = len(nums)
    
    # 外层循环枚举最长边 c=nums[k]
    for k in range(n - 1, 1, -1):
        c = nums[k]
        # 相向双指针：i 指向开头，j 指向 k-1 位置
        i = 0
        j = k - 1
        
        while i < j:
            # 如果 nums[i] + nums[j] > c，说明能构成三角形
            if nums[i] + nums[j] > c:
                # 由于数组是有序的，nums[j] 与下标 i' 在 [i, j-1] 中的任何 nums[i'] 相加，都是 >c 的
                # 因此直接找到了 j-i 个合法方案
                count += j - i
                # 移动右指针
                j -= 1
            else:
                # 如果 nums[i] + nums[j] <= c，由于数组是有序的，
                # nums[i] 与下标 j' 在 [i+1, j] 中的任何 nums[j'] 相加，都是 <=c 的
                # 因此后面无需考虑 nums[i]，将 i 加一
                i += 1
    
    return count


def triangle_number_method2(nums):
    """
    方法二：暴力法
    
    时间复杂度：O(n^3)
    空间复杂度：O(1)
    
    算法思路：
    1. 先对数组进行排序
    2. 三重循环遍历所有可能的三元组
    3. 对于每个三元组，检查是否满足三角形条件
    
    参数：
        nums: 包含非负整数的数组
    
    返回：
        能组成三角形的三元组个数
    """
    if len(nums) < 3:
        return 0
    
    # 先对数组进行排序
    nums.sort()
    count = 0
    n = len(nums)
    
    # 三重循环遍历所有可能的三元组
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                # 由于已排序，nums[i] <= nums[j] <= nums[k]
                # 只需检查 nums[i] + nums[j] > nums[k]
                if nums[i] + nums[j] > nums[k]:
                    count += 1
    
    return count


def print_all_triangles(nums):
    """
    打印所有有效的三角形组合
    
    参数：
        nums: 包含非负整数的数组
    """
    if len(nums) < 3:
        print("数组长度小于3，无法组成三角形")
        return
    
    nums_sorted = sorted(nums)
    triangles = []
    n = len(nums_sorted)
    
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                if nums_sorted[i] + nums_sorted[j] > nums_sorted[k]:
                    triangles.append((nums_sorted[i], nums_sorted[j], nums_sorted[k]))
    
    print(f"有效的三角形组合（共 {len(triangles)} 个）：")
    for idx, triangle in enumerate(triangles, 1):
        print(f"  {idx}. {triangle}")


def generate_test_data(n, min_val=1, max_val=1000):
    """
    生成测试数据用于性能测试
    
    参数：
        n: 数组长度
        min_val: 最小值（默认1，避免0）
        max_val: 最大值（默认1000）
    
    返回：
        生成的测试数组
    """
    return [random.randint(min_val, max_val) for _ in range(n)]


def time_method(method_func, nums, method_name):
    """
    计时执行方法并返回结果和时间
    
    参数：
        method_func: 要执行的方法函数
        nums: 输入数组
        method_name: 方法名称
    
    返回：
        (结果, 执行时间(秒))
    """
    start_time = time.perf_counter()
    result = method_func(nums.copy())
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return result, elapsed_time


def main():
    """
    主函数：测试示例并比较效率
    """
    # 示例1
    print("示例1：")
    nums1 = [2, 2, 3, 4]
    print(f"输入: nums = {nums1}")
    
    result1, time1 = time_method(triangle_number_method1, nums1, "方法一")
    result2, time2 = time_method(triangle_number_method2, nums1, "方法二")
    
    print(f"方法一: 结果={result1}, 耗时={time1*1000:.4f}ms")
    print(f"方法二: 结果={result2}, 耗时={time2*1000:.4f}ms")
    
    print(f"\n解释：有效的组合是：")
    print_all_triangles(nums1)
    
    # 示例2
    print("示例2：大数据量测试（用于比较效率）")
    
    # 生成测试数据
    size = 200  # 测试规模：200个元素
    print(f"\n测试规模: {size} 个元素")
    nums2 = generate_test_data(size, min_val=1, max_val=1000)
    print(f"输入数组长度: {len(nums2)}")
    print(f"输入数组范围: [{min(nums2)}, {max(nums2)}]")
    
    # 测试方法一
    result1, time1 = time_method(triangle_number_method1, nums2, "方法一")
    print(f"方法一: 结果={result1}, 耗时={time1*1000:.4f}ms")
    
    # 测试方法二
    result2, time2 = time_method(triangle_number_method2, nums2, "方法二")
    print(f"方法二: 结果={result2}, 耗时={time2*1000:.4f}ms")
    
    # 验证结果一致性
    if result1 == result2:
        print(f"两种方法结果一致: {result1}")
    else:
        print(f"方法结果不一致！方法一={result1}, 方法二={result2}")


if __name__ == "__main__":
    main()

