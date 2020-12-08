# LeetCode 刷題記錄
題目順序無規律，My Code皆使用Python3語言。
標題開頭表示該題的難易度分類：(E)為Easy、(M)為Medium、(H)為Hard
單純是我個人解題當下的紀錄，不會是唯一或最佳解。

## (M) 39. Combination Sum
https://leetcode.com/problems/combination-sum/

### Description
給一組**candidates**(list of int)和**target**(int)，求使用candidates內數字組合，讓總和等於target的所有組合。
- candidates內的數字皆不同
- candidates內的數字可以重複使用無限次
- candidates內的數字介於1~200
- candidates的長度介於1~30
- target介於1~500

### Example
```
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.
```

### My Idea
假設candidates的其中一個數為n，可以延伸出一個相同設定的子問題：**新candidates(candidates拿掉n)與新target(target-n)的求解**。將這個子問題的解全部加入n就是原題目的解。所以此題可以使用遞迴解法，考慮邊界條件並定義終止條件為```min(candidates) == target```，此時子問題就只會有[min(candidates)]這個解。

### My Code
```python=
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort(reverse=True)
        return self.get_combinations(candidates, target)
    
    def get_combinations(self, candidates, target):
        min_c = candidates[-1]
        if target < min_c:
            return []
        elif target == min_c:
            return [[min_c]]
        else:
            ans_list = []
            for i, c in enumerate(candidates):
                if c > target:
                    continue
                elif c == target:
                    ans_list.append([c])
                else:
                    left_candidates = candidates[i:]
                    left_target = target - c
                    combinations = self.get_combinations(left_candidates, left_target)
                    for combination in combinations:
                        new_ans = [c] + combination
                        ans_list.append(new_ans)
            return ans_list
```

### Submission
![](https://i.imgur.com/6U4GBSH.png =400x)

## (M) 532. K-diff Pairs in an Array
https://leetcode.com/problems/k-diff-pairs-in-an-array/

### Description
給定一組**nums**(list of int)和**k**(int)，求使用nums內任兩個數字配對，可以形成的不重複**k-diff配對**(兩數字相差k)總數量。
- 針對一組**k-diff配對**```(nums[i], nums[j])```，符合以下條件
    - ```0 <= i, j < nums.length```
    - ```i != j```
    - ```|nums[i] - nums[j]| == k```
- 1 <= nums.length <= 104
- 107 <= nums[i] <= 107
- 0 <= k <= 107

### Example
```
Input: nums = [3,1,4,1,5], k = 2
Output: 2
Explanation: There are two 2-diff pairs in the array, (1, 3) and (3, 5).
Although we have two 1s in the input, we should only return the number of unique pairs.
```

### My Idea
由於是要兩兩配對，看是否符合k-diff的條件，**可以將nums中每個數字加上k，形成target_nums**，然後使用**雙指針**分別對著nums與target_nums，依序檢查所有符合的配對。這裡有兩點需要注意
1. 由於配對不能重複，要在前面先排除nums中重複出現的數字
2. special case : 當k等於0時，改為去檢查出現多少重複數字

### My Code
```python=
class Solution:
    def findPairs(self, nums: List[int], k: int) -> int:
        count = 0
        nums.sort()
        if k == 0: # special case
            i = 0
            tmp_n = nums[0] - 1
            max_i = len(nums) - 1
            while i < max_i:
                n = nums[i]
                if n != tmp_n and n == nums[i+1]:
                    count += 1
                    tmp_n = n
                i += 1
        else: # normal case
            nums1 = list(dict.fromkeys(nums)) # remove all duplicated numbers
            nums2 = [n + k for n in nums1]
            i1, i2 = 0, 0
            len1, len2 = len(nums1), len(nums2)
            while i1 < len1 and i2 < len2:
                n1 = nums1[i1]
                n2 = nums2[i2]
                if n1 == n2:
                    count += 1
                    i1 += 1
                    i2 += 1
                elif n1 < n2:
                    i1 += 1
                else:
                    i2 += 1
        return count
```

### Submission
![](https://i.imgur.com/rU1qAZP.png =400x)

## (M) 1288. Remove Covered Intervals
https://leetcode.com/problems/remove-covered-intervals/

### Description
給定一個**intervals列表**，要**移除所有被其他大interval包含住(covered)的小interval**，只剩下互不包含的intervals，然後回傳剩下的個數。
- interval ```[a,b)``` is **covered** by interval ```[c,d)``` if and only if ```c <= a``` and ```b <= d``` .
- 1 <= intervals.length <= 1000
- intervals[i].length == 2
- 0 <= intervals[i][0] < intervals[i][1] <= 10^5
- 所有intervals皆不重複

### Example
```
Input: intervals = [[1,4],[3,6],[2,8]]
Output: 2
Explanation: Interval [3,6] is covered by [2,8], therefore it is removed.
```

### My Idea
先將interval列表由小到大sort，確保遍歷時，先取的interval```[a, b]```和後取的interval```[c, d]```必定符合**a不大於c**。所以任兩個intervals比對時，只需要檢查**d是否小於b**即可確認是否有包含的狀況。
要注意一個**特殊狀況：當a等於c**，如果d大於b，表示先取的```[a, b]```是被包含在後取的```[c, d]```中。

### My Code
```python=
class Solution:
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort()
        covered_count = 0
        i = 0
        len_intervals = len(intervals)
        while i < len_intervals - 1:
            a, b = intervals[i]
            for j in range(i + 1, len_intervals):
                c, d = intervals[j]
                if d > b:
                    if a == c: # special case, [a, b] is covered by [c, d]
                        covered_count += 1
                    break
                else:
                    covered_count += 1
            i = j
        return len_intervals - covered_count
```

### Submission
![](https://i.imgur.com/cnCFSMX.png =400x)

## (E) 1009. Complement of Base 10 Integer
https://leetcode.com/problems/complement-of-base-10-integer/

### Description
給定一個非負整數**N**(int)，可以將其以binary表示，假設為binary_N(str)，ex. N=5則binary_N='101'。求此binary_N的**complement**(binary inverse)，ex.'101'的complement為'010'。回傳以此complement表示的十進位整數(int)。

### Example
```
Input: 5
Output: 2
Explanation: 5 is "101" in binary, with complement "010" in binary, which is 2 in base-10.
```

### My Idea
此題很直觀，將N轉成binary後，1換成0，0換成1，再轉回int即可。

### My Code
```python=
class Solution:
    def bitwiseComplement(self, N: int) -> int:
        bin_n = f'{N:b}'
        bin_dict = {'0': '1', '1': '0'}
        complement_n = ''
        for b in bin_n:
            complement_n += bin_dict[b]
        return int(complement_n, 2)
```

### Submission
![](https://i.imgur.com/xzfWm15.png =400x)

## (M) 701. Insert into a Binary Search Tree
https://leetcode.com/problems/insert-into-a-binary-search-tree/

### Description
給定一個binary search tree(BST)的節點列表**root**(list of int)以及一個要插入的**value**(int)，回傳插入完成後的BST節點列表。
- 保證value不存在原來的BST中
- 可能存在多種插入方式，回傳任一種結果即可
- BST中的節點數量介於0~10^4
- BST中的節點數值與要插入的value皆介於-10^8~10^8，且無重複

### Example
![](https://i.imgur.com/1gGKsiL.png)
```
Input: root = [4,2,7,1,3], val = 5
Output: [4,2,7,1,3,5]
```

### My Idea
為要插入的value建立新的node為new_node，然後從root這個node開始遍歷整棵樹，找到正確的位置插入，再回傳root。假如root一開始就是空的，則直接回傳new_node。
遍歷方式：如果value大於node.val，取node.right為下一個node；如果value小於node.val，取node.left為下一個node，當下一個node為空(None)時，就在此插入new_node。

### My Code
```python=
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        new_node = TreeNode(val)
        if root is None:
            return new_node
        else:
            node = root
            while True:
                node_val = node.val
                if val > node_val:
                    if node.right is None:
                        node.right = new_node
                        break
                    else:
                        node = node.right
                else:
                    if node.left is None:
                        node.left = new_node
                        break
                    else:
                        node = node.left
            return root
```

### Submission
![](https://i.imgur.com/zoNYlsK.png =400x)

## (M) 61. Rotate List
https://leetcode.com/problems/rotate-list/

### Description
給定一個**linked list**和非負整數**k**(int)，要將linked_list的最後一個node移到第一個node，連續進行k次，回傳最後結果。

### Example
```
Input: 1->2->3->4->5->NULL, k = 2
Output: 4->5->1->2->3->NULL
Explanation:
rotate 1 steps to the right: 5->1->2->3->4->NULL
rotate 2 steps to the right: 4->5->1->2->3->NULL
```

### My Idea
根據題目敘述與範例，其實是**將linked list切成左右兩個區塊，然後位置互換而已，這兩區塊內的連接順序並不會改變**。所以只要找到切分成左右區塊的位置，將各區塊的邊界node.next修改即可。
由於k有可能大於linked list的長度，因此先遍歷一次linked list確認長度，然後將k對其取餘數，就能得知要切分的位置。

### My Code
```python=
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if head is None:
            return None
        list_len = 0
        node = head
        while True:
            list_len += 1
            if node.next is None:
                last_node = node
                break
            else:
                node = node.next
        k = k % list_len # make sure k is smaller than linked list length
        if k == 0:
            return head
        else:
            rotate_i = list_len - k - 1
            node = head
            for i in range(rotate_i):
                node = node.next
            new_head = node.next
            node.next = None
            last_node.next = head
            return new_head
```

### Submission
![](https://i.imgur.com/9VJxOfV.png =400x)

## (M) 449. Serialize and Deserialize BST
https://leetcode.com/problems/serialize-and-deserialize-bst/

### Description
給定一個**二分搜索樹BST**，設計一個serialize(序列化)與一個deserialize(反序列化)的方法，讓BST可以序列化成str後再反序列化為原本的BST。
- 樹的Node數量介於0~10^4
- 0 <= Node.val <= 104
- 保證是BST

### Example
```
Input: root = [2,1,3]
Output: [2,1,3]
```

### My Idea
利用BST的特性，賦予每個node一個獨立的ID，與其val一起保存在str中。設root的ID為1，則對於**ID為n的node**有以下特性：**left node ID = n * 2**、**right node ID = n * 2 + 1**。
根據此規則序列化所有node的資訊在str中，反序列化時依照此規則生成並連接所有node即可。

### My Code
```python=
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root: TreeNode) -> str:
        """Encodes a tree to a single string.
        """
        if root is None:
            return ''
        node_id = 1
        code = self.node_encode(root, node_id)
        return code
        
    def node_encode(self, node: TreeNode, node_id) -> str:
        code = f'{node_id},{node.val}/'
        left_id = node_id * 2
        right_id = left_id + 1
        if node.left is not None:
            code += self.node_encode(node.left, left_id)
        if node.right is not None:
            code += self.node_encode(node.right, right_id)
        return code

    def deserialize(self, data: str) -> TreeNode:
        """Decodes your encoded data to tree.
        """
        if data == '':
            return None
        node_list = data.split('/')[:-1]
        self.node_dict = {}
        for node_str in node_list:
            node_id, node_val = node_str.split(',')
            self.node_dict[int(node_id)] = node_val
        root = self.get_node(1)
        return root

    def get_node(self, node_id) -> TreeNode:
        if node_id in self.node_dict:
            node = TreeNode(self.node_dict[node_id])
            left_id = node_id * 2
            right_id = left_id + 1
            node.left = self.get_node(left_id)
            node.right = self.get_node(right_id)
        else:
            node = None
        return node

# Your Codec object will be instantiated and called as such:
# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# tree = ser.serialize(root)
# ans = deser.deserialize(tree)
# return ans
```

### Submission
![](https://i.imgur.com/6WptThR.png =400x)

## (M) 452. Minimum Number of Arrows to Burst Balloons
https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/

### Description
二維空間中散布多個氣球，每個氣球給定```x_start```與```x_end```，代表氣球的寬度，而高度不重要。可以從地面垂直向上發射弓箭，如果從```x```處發射，則所有符合```x_start <= x <= x_end```的氣球均會被射破。
題目給定所有**氣球列表points**(list, ```points[i] = [x_start, x_end]```)，求**最少需要發射多少弓箭才能把所有氣球都射破**。
- 氣球數介於0~10^4
- -2^31 <= x_start < x_end <= 2^31 - 1

### Example
```
Input: points = [[10,16],[2,8],[1,6],[7,12]]
Output: 2
Explanation: One way is to shoot one arrow for example at x = 6 (bursting the balloons [2,8] and [1,6]) and another arrow at x = 11 (bursting the other two balloons).
```

### My Idea
這題目還滿有趣的，基本就是要盡量讓x座標上有重疊的氣球用一箭就射破。一樣先將氣球座標由小到大排序，然後從x最小(最左邊)的氣球開始遍歷，拿當下的氣球([x_start, x_end])和下一個氣球([x1_start, x1_end])比較座標範圍，當遇到一個完全無交集的新氣球(x_end < x1_start)，表示之前的氣球都能用一支箭一起射破。注意遍歷完的最後一組氣球，也需要一支箭射破。

### My Code
```python=
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if len(points) == 0:
            return 0
        points.sort()
        balloon_n = len(points)
        i = 0
        arrow_count = 0
        while i < balloon_n - 1:
            balloon = points[i]
            x_start, x_end = balloon
            for j in range(i + 1, balloon_n):
                next_balloon = points[j]
                x1_start, x1_end = next_balloon
                if x_start < x1_start:
                    if x_end >= x1_end: # smaller balloon
                        break
                    elif x_end < x1_start: # no intersection balloon
                        arrow_count += 1 # previous balloons need an arrow
                        break
            i = j
        arrow_count += 1 # last arrow shot
        return arrow_count
```

### Submission
![](https://i.imgur.com/j7A1OjD.png =400x)

## (M) 316. Remove Duplicate Letters
https://leetcode.com/problems/remove-duplicate-letters/

### Description
給定一個字串**s**，移除重複出現的字母，讓每個字母最多出現一次，並且要確保，回傳所有可能結果中**字典序排最小**的那個結果。
- s的長度介於1~10^4
- s只包含小寫英文字母

### Example
```
Input: s = "bcabc"
Output: "abc"
```
```
Input: s = "cbacdcbc"
Output: "acdb"
```

### My Idea
先列出所有出現的字母，接著sort讓他們**按照字典序排序**。**紀錄每個字母最後出現的位置**(在原始字串中的index)，以利後面做比較。紀錄一個offset，代表最新一個取的字母所在位置，之後**只能從這個offset之後開始取其他字母**。
從待排字母中(一開始所有出現的字母都是待排字母)，字典序最前面的字母開始，**檢查此字母的位置(index)是否小於全部其餘待排字母的最後出現位置**。如果符合就取出這個字母到答案字串中，如果不符合就要往下一個待排字母找，一直到全部字母都被取完。

### My Code
```python=
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        char_set = set(s)
        char_last_idx = dict()
        for idx, char in enumerate(s):
            char_last_idx[char] = idx

        chars = list(char_set)
        chars.sort() # sort to get lexicographical order
        char_n = len(chars)
        offset = 0 # offset of original s
        new_s = ''
        for i in range(char_n):
            for c in chars:
                check_fail = False
                c_idx = s.find(c) # character index of modified s
                check_idx = c_idx + offset # character index of original s
                # check if all other needed characters exist after this character
                for check_c in chars:
                    if check_idx > char_last_idx[check_c]:
                        check_fail = True
                        break
                # if check pass, add charater to new_s and update offset and modify s
                if not check_fail:
                    new_s += c
                    offset = check_idx
                    s = s[c_idx:]
                    break
            chars.remove(c)
        return new_s
```

### Submission
![](https://i.imgur.com/NOsTkJ3.png =400x)

## (E) 1290. Convert Binary Number in a Linked List to Integer
https://leetcode.com/problems/convert-binary-number-in-a-linked-list-to-integer/

### Description
給定一個singly-linked list的頭節點**head**，**每個節點的value是0或1**，整個linked list包含一個數字的二進位表示，回傳此數字的十進位表示。
- linked list不會是空的
- 節點數不超過30

### Example
```
Input: head = [1,0,1]
Output: 5
Explanation: (101) in base 2 = (5) in base 10
```

### My Idea
此題非常簡單，只要遍歷linked list，依序存下每個節點的value，最後轉成int回傳即可。

### My Code
```python=
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        n_str = ''
        node = head
        while node is not None:
            n_str += str(node.val)
            node = node.next
        return int(n_str, 2)
```

### Submission
![](https://i.imgur.com/28ZQuei.png =400x)

## (M) 147. Insertion Sort List
https://leetcode.com/problems/insertion-sort-list/

### Description
給定一個linked list，使用插入排序(insertion sort)，回傳排序後的結果。

### Example
```
Input: 4->2->1->3
Output: 1->2->3->4
```

### My Idea
先參考題目頁面對插入排序的解說。創立一個新的空linked list，然後對舊的linked list遍歷，一個個依序插入到新的linked list中。

### My Code
```python=
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def insertionSortList(self, head: ListNode) -> ListNode:
        node = head
        new_head = None
        while node is not None:
            new_node = ListNode(node.val)
            new_head = self.insert_node(new_node, new_head)
            node = node.next
        return new_head

    def insert_node(self, node: ListNode, head: ListNode) -> ListNode:
        if head is None:
            node.next = None
            return node
        else:
            list_node = head
            prev_node = None
            while list_node is not None:
                if node.val <= list_node.val:
                    node.next = list_node
                    if prev_node is None:
                        head = node
                    else:
                        prev_node.next = node
                    break
                else:
                    prev_node = list_node
                    list_node = list_node.next
            if list_node is None:
                node.next = None
                prev_node.next = node
            return head
```

### Submission
![](https://i.imgur.com/WIsd6Ku.png =400x)

## (E) 859. Buddy Strings
https://leetcode.com/problems/buddy-strings/

### Description
給定兩個字串**A**、**B**，檢查是否可以**將A中的某兩個字符交換後等於B**，如果可以回傳True，否則回傳False。

### Example
```
Input: A = "ab", B = "ba"
Output: true
Explanation: You can swap A[0] = 'a' and A[1] = 'b' to get "ba", which is equal to B.
```

### My Idea
遍歷兩個字串，記錄下不同的字符，再檢查是否互換後相等即可。有幾個特殊情況如下：
1. 兩字串長度不同，即可回傳False
2. 兩字串一模一樣，改為檢查字串中是否有重複字符
3. 兩字串不同的字符超過2個，即可回傳False

### My Code
```python=
class Solution:
    def buddyStrings(self, A: str, B: str) -> bool:
        len_a = len(A)
        len_b = len(B)
        if len_a != len_b:
            return False
        elif A == B:
            if len(set(A)) == len_a: # no duplicated char
                return False
            else:
                return True
        else:
            diff_a = []
            diff_b = []
            diff_count = 0
            for i in range(len_a):
                a = A[i]
                b = B[i]
                if a != b:
                    diff_a.append(a)
                    diff_b.append(b)
                    diff_count += 1
                if diff_count > 2: # more than two diff char, can't handle in one swap
                    return False
            if set(diff_a) == set(diff_b):
                return True
            else:
                return False
```

### Submission
![](https://i.imgur.com/fLH5Db5.png =400x)

## (E) 1446. Consecutive Characters
https://leetcode.com/problems/consecutive-characters/

### Description
給定一個字串**s**，計算其power並回傳。power的定義為：字串所有只包含單一字母的substring中，最長的長度。
- s的長度介於1~500
- s只包含小寫英文字母

### Example
```
Input: s = "leetcode"
Output: 2
Explanation: The substring "ee" is of length 2 with the character 'e' only.
```

### My Idea
遍歷一次字串，比較並記錄最長的連續相同字母出現次數即可。

### My Code
```python=
class Solution:
    def maxPower(self, s: str) -> int:
        max_power = 1
        last_letter = None
        power = 0
        for c in s:
            if c != last_letter:
                if power > max_power:
                    max_power = power
                power = 1
            else:
                power += 1
            last_letter = c
        if power > max_power:
            max_power = power
        return max_power
```

### Submission
![](https://i.imgur.com/8AnuqnQ.png =400x)

## (M) 213. House Robber II
https://leetcode.com/problems/house-robber-ii/

### Description
假設你是一個職業小偷，要在**一群房屋**中偷錢，每個房屋中都有數量不一的金錢，而所有房屋的位置以**環狀相連形成一個圓**(第一間房屋與最後一間房屋相鄰)。另外，每兩個相鄰房屋有一個共同的警報器，當這**兩間相鄰房屋都被闖入偷竊**時，會觸發警報並呼叫警察。
給定一個非負整數的陣列**nums**，裡面的數字依序代表每個房屋有的錢財，求在**不觸發警報**的情況下，你**最多可以偷到多少錢**。
- 1 <= nums.length <= 100
- 0 <= nums[i] <= 1000

### Example
```
Input: nums = [2,3,2]
Output: 3
Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.
```

### My Idea
使用**Dynamic Programming**(動態規劃)的概念，建立一個dp_table儲存各個子問題的解，**根據小範圍子問題的解得到更大範圍子問題的解**，最後得到原始問題的解。
將**子問題**設定為：**給定數字陣列，相鄰的數字不能都取，求最大總和**。在```dp_table[i][j]```儲存```nums[i:j]```的解，則每個子問題```dp_table[i][j]```的解，是從下列兩個選項中取**較大值**：**選項一**```nums[i] + dp_table[i+2][j]```(取第一個數，不取第二個數)、**選項二**```nums[i+1] + dp_table[i+3][j]```(取第二個數，不取第一、三個數)
上述子問題沒有考慮頭尾相鄰，但原始問題有頭尾相鄰，所以**原始解的選項一也不能取最後一個數**。

### My Code
```python=
class Solution:
    def rob(self, nums: List[int]) -> int:
        house_n = len(nums)
        self.moneys = nums
        self.dp_table = [[None] * (house_n + 1) for i in range(house_n + 1)]
        if house_n == 1:
            return nums[0]
        else:
            money1 = nums[0] + self.sub_rob(2, house_n - 1)
            money2 = self.sub_rob(1, house_n)
            print(money1, money2)
            if money1 >= money2:
                return money1
            else:
                return money2

    def sub_rob(self, i, j):
        ans = self.dp_table[i][j]
        if ans is not None:
            return ans
        else:
            nums = self.moneys[i:j]
            print(nums)
            len_n = len(nums)
            if len_n == 0:
                ans = 0
            elif len_n == 1:
                ans = nums[0]
            elif len_n == 2:
                if nums[0] >= nums[1]:
                    ans = nums[0]
                else:
                    ans = nums[1]
            else:
                money1 = nums[0] + self.sub_rob(i + 2, j)
                money2 = nums[1] + self.sub_rob(i + 3, j)
                if money1 >= money2:
                    ans = money1
                else:
                    ans = money2
            self.dp_table[i][j] = ans
            return ans
```

### Submission
![](https://i.imgur.com/UMHmaVy.png =400x)

## (M) 310. Minimum Height Trees
https://leetcode.com/problems/minimum-height-trees/

### Description
給定一棵樹的node數量**n**(ID為0 ~ n-1)，還有n-1條**edges**表示所有node的連接關係(```edges[i] = [n1, n2]```，代表node n1與node n2有連接)。此時，選擇任意一個node當成root皆可形成一棵樹，並且樹的高度為h。求在所有node選項中，能**讓樹高h為最小的所有root**。
- node數量介於1~2*10^4
- edge數量必為n-1
- n1 != n2
- edges中沒有重複的連接
- 給定的數據保證能形成一棵樹

### Example
![](https://i.imgur.com/JsidRpr.png)
```
Input: n = 4, edges = [[1,0],[1,2],[1,3]]
Output: [1]
Explanation: As shown, the height of the tree is 1 when the root is the node with label 1 which is the only MHT.
```

### My Idea
由於題目保證能形成一棵樹，如果**要讓高度最小，表示所有子樹要盡量平衡**(不論是二叉到N叉樹)。假設所有子樹高度平衡，要找root的話，就可以**從葉子節點開始，一層一層拔掉往上找，最後剩下的就會是root節點**。
先遍歷所有edges，記錄下每個node的連接列表。接著每一輪都找出葉子節點(連接數為1)，拔除後檢查其父節點是否變為葉子節點，是的話就加入下一輪的葉子節點名單，重複動作直到剩下1或2個節點，就是root了。

### My Code
```python=
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n <= 2:
            return [i for i in range(n)]

        node_edges = {i: [] for i in range(n)}
        for n1, n2 in edges:
            node_edges[n1].append(n2)
            node_edges[n2].append(n1)

        node_n = n
        leaves = []
        for n_id, connect_list in node_edges.items():
            if len(connect_list) == 1:
                leaves.append(n_id)

        while node_n > 2:
            node_n -= len(leaves)
            new_leaves = []
            for n_id in leaves:
                connect_id = node_edges[n_id][0]
                node_edges[connect_id].remove(n_id)
                if len(node_edges[connect_id]) == 1:
                    new_leaves.append(connect_id)
            leaves = new_leaves
        
        return leaves
```

### Submission
![](https://i.imgur.com/9onD5oT.png =400x)

## (E) 1217. Minimum Cost to Move Chips to The Same Position
https://leetcode.com/problems/minimum-cost-to-move-chips-to-the-same-position/

### Description
現在有n個圓片分布在一維座標空間上，給定**position**列表表示這些圓片的分布位置，```position[i]```代表**第i個圓片的位置**。對於```position[i]```圓片，有移動方式如下：
- ```position[i] + 2``` or ```position[i] - 2``` with ```cost = 0```
- ```position[i] + 1``` or ```position[i] - 1``` with ```cost = 1```

求將**所有圓片移動到相同位置上，最少需要花多少cost**。
- ```1 <= position.length <= 100```
- ```1 <= position[i] <= 10^9```

### Example
![](https://i.imgur.com/SG4Wcxy.png)
```
Input: position = [1,2,3]
Output: 1
Explanation: First step: Move the chip at position 3 to position 1 with cost = 0.
Second step: Move the chip at position 2 to position 1 with cost = 1.
Total cost is 1.
```

### My Idea
從移動規則可以看出，只有奇數與偶數之間的位置轉換會需要cost 1。所以只要計算原本奇數與偶數位置的圓片個數，將數量少的那一方移動到另一方的任一位置上即可。

### My Code
```python=
class Solution:
    def minCostToMoveChips(self, position: List[int]) -> int:
        odd_count = 0
        even_count = 0
        for x in position:
            if x % 2 == 0:
                even_count += 1
            else:
                odd_count += 1
        if odd_count > even_count:
            return even_count
        else:
            return odd_count
```

### Submission
![](https://i.imgur.com/Ww3TpKQ.png =400x)

## (M) 1283. Find the Smallest Divisor Given a Threshold
https://leetcode.com/problems/find-the-smallest-divisor-given-a-threshold/

### Description
給定**nums**(list of int)與**threshold**(int)，選擇一個**正整數divisor去除nums中的每個數字，再將所有除後結果相加**，得到最後結果。求使最後結果**不大於threshold**的**最小divisor**。
- ```1 <= nums.length <= 5 * 10^4```
- ```1 <= nums[i] <= 10^6```
- ```nums.length <= threshold <= 10^6```

### Example
```
Input: nums = [1,2,5,9], threshold = 6
Output: 5
Explanation: We can get a sum to 17 (1+2+5+9) if the divisor is 1. 
If the divisor is 4 we can get a sum to 7 (1+1+2+3) and if the divisor is 5 the sum will be 5 (1+1+1+2). 
```

### My Idea
解題方式不難，就是依序使用不同的divisor計算除後總和，直到找到解答。此題重點是要使用 **binary search** 來加快尋找divisor的效率，否則就會超時。

### My Code
```python=
class Solution:
    def smallestDivisor(self, nums: List[int], threshold: int) -> int:
        num_n = len(nums)
        if num_n == threshold:
            return nums[-1]

        self.nums = nums
        left = 0
        right = 10
        while self.divide_sum(right) > threshold:
            left = right
            right *= 10
        mid = int((left + right) / 2)
        while left < mid:
            div_sum = self.divide_sum(mid)
            if div_sum > threshold:
                left = mid
                mid = int((left + right) / 2)
            else:
                right = mid
                mid = int((left + right) / 2)
        return right

    def divide_sum(self, divisor):
        div_sum = 0
        for num in self.nums:
            if num <= divisor:
                div_sum += 1
            else:
                div_sum += ceil(num / divisor)
        return div_sum
```

### Submission
![](https://i.imgur.com/bRWfc5V.png =400x)

## (E) 938. Range Sum of BST
https://leetcode.com/problems/range-sum-of-bst/

### Description
給定一棵**binary search tree**的**root**，以及上下限**low**、**high**，求整棵樹在上下限之間的所有value的總和。
- node數量介於 1 ~ 2*10^4
- ```1 <= Node.val <= 10^5```
- ```1 <= low <= high <= 10^5```
- 所有node的value皆不同

### Example
![](https://i.imgur.com/x9QPHfb.png =300x)
```
Input: root = [10,5,15,3,7,null,18], low = 7, high = 15
Output: 32
```

### My Idea
此題用遞迴遍歷整棵樹，將所有value加總即可。只需要注意，如果當前node的值已經低於下限，就不用遍歷左子樹；如果當前node的值已經高於上限，就不用遍歷右子樹。

### My Code
```python=
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        self.low = low
        self.high = high
        return self.compute_sum(root)

    def compute_sum(self, node):
        if node is None:
            return 0
        tree_sum = 0
        node_val = node.val
        if node_val < self.low:
            tree_sum += self.compute_sum(node.right)
        elif node_val > self.high:
            tree_sum += self.compute_sum(node.left)
        else:
            tree_sum += node_val
            tree_sum += self.compute_sum(node.left)
            tree_sum += self.compute_sum(node.right)
        return tree_sum
```

### Submission
![](https://i.imgur.com/pGpg8C7.png =400x)

## (M) 47. Permutations II
https://leetcode.com/problems/permutations-ii/

### Description
給定一串數字列表**nums**，其中**可能有重複**的數字，回傳這些數字能組成的**所有獨特排列方式**(回傳值中的排列不限順序但不能重複)。
- ```1 <= nums.length <= 8```
- ```-10 <= nums[i] <= 10```

### Example
```
Input: nums = [1,1,2]
Output: [[1,1,2], [1,2,1], [2,1,1]]
```

### My Idea
數字列表中的每個獨特數字都可以產生子問題：當取出此數字n當開頭，分別接上剩下數字列表能產生的所有排列，就會是以n為首的所有排列。可以用遞迴的方式計算出子問題的所有組合，需要注意遞迴的中止條件(當剩下數字列表中只有一個獨特數字)。遍歷所有子問題即可得到最終解。

### My Code
```python=
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        return self.get_permutations(nums)

    def get_permutations(self, nums):
        if len(set(nums)) == 1:
            return [[nums[0] for i in range(len(nums))]]
        permutations = []
        used_nums = []
        for i, num in enumerate(nums):
            if num not in used_nums:
                tmp = nums.copy()
                tmp.pop(i)
                sub_permutations = self.get_permutations(tmp)
                for sub_permutation in sub_permutations:
                    permutations.append([num] + sub_permutation)
                used_nums.append(num)
        return permutations
```

### Submission
![](https://i.imgur.com/KodbWX0.png =400x)

## (E) 832. Flipping an Image
https://leetcode.com/problems/flipping-an-image/

### Description
給定一個 binary matrix **A**，先將其作**水平翻轉**，再做**二進制反轉**，回傳結果。
- 水平翻轉 : 同一個row左右相反，ex. [1, 1, 0] -> [0, 1, 1]
- 二進制反轉 : 1與0的轉換，ex. [0, 1, 1] -> [1, 0, 0]
- ```1 <= A.length = A[0].length <= 20```
- ```0 <= A[i][j] <= 1```

### Example
```
Input: [[1,1,0],[1,0,1],[0,0,0]]
Output: [[1,0,0],[0,1,0],[1,1,1]]
Explanation: First reverse each row: [[0,1,1],[1,0,1],[0,0,0]].
Then, invert the image: [[1,0,0],[0,1,0],[1,1,1]]
```

### My Idea
此題很簡單，照題意依序轉換即可。

### My Code
```python=
class Solution:
    def flipAndInvertImage(self, A: List[List[int]]) -> List[List[int]]:
        fliped = []
        invert_dict = {0: 1, 1: 0}
        for row in A:
            row.reverse()
            inverted_row = [invert_dict[val] for val in row]
            fliped.append(inverted_row)
        return fliped
```

### Submission
![](https://i.imgur.com/F9HPcqF.png =400x)

## (M) 845. Longest Mountain in Array
https://leetcode.com/problems/longest-mountain-in-array/

### Description
給定數字列表**A**(array of int)，求其中包含的**最長mountain**(下面會定義)**的長度**，如果沒有mountain則回傳0。
- ```0 <= A.length <= 10000```
- ```0 <= A[i] <= 10000```

mountain定義如下 :
- 是A的subarray(以下簡稱B，可以等於A)
- ```B.length >= 3```
- 其中有一個```i```符合```0 < i < B.length - 1```，使得```B[0] < B[1] < ... B[i-1] < B[i] > B[i+1] > ... > B[B.length - 1]```

### Example
```
Input: [2,1,4,7,3,2,5]
Output: 5
Explanation: The largest mountain is [1,4,7,3,2] which has length 5.
```

### My Idea
根據題意，要形成mountain需要有左邊的上坡與右邊的下坡，維護一個最大mountain長度的變數(```max_len```)，設定一個上下坡狀態指示變數(```mountain_up```)。
當處於上坡狀態(```mountain_up為True```)時，當前數值要大於前一個數值，如果小於前一個數值，就轉換為下坡狀態(```mountain_up為False```)；當處於下坡狀態時，當前數值要小於前一個數值，一旦違反，表示形成了一個mountain，就更新```max_len```並重置狀態，持續此流程直到結束。
要注意，如果是平坡(數值與前一個相等)，也不能形成mountain，要直接重置狀態。

### My Code
```python=
class Solution:
    def longestMountain(self, A: List[int]) -> int:
        if len(A) <= 2:
            return 0
        
        max_len = 0
        mountain_up = True
        prev_n = A[0]
        mountain_len = 0
        for n in A[1:]:
            if mountain_up:
                if n > prev_n:
                    mountain_len += 1
                elif n < prev_n:
                    if mountain_len > 0:
                        mountain_up = False
                        mountain_len += 1
                else:
                    mountain_len = 0
            else:
                if n < prev_n:
                    mountain_len += 1
                else:
                    mountain_len += 1
                    if mountain_len > max_len:
                        max_len = mountain_len
                    if n > prev_n:
                        mountain_len = 1
                    else:
                        mountain_len = 0
                    mountain_up = True
            prev_n = n
        
        if not mountain_up:
            mountain_len += 1
            if mountain_len > max_len:
                max_len = mountain_len

        return max_len
```

### Submission
![](https://i.imgur.com/vyuRYMB.png =400x)

## (M) 858. Mirror Reflection
https://leetcode.com/problems/mirror-reflection/

### Description
在2D空間中，有一個正方形的房間，四邊都是鏡子，然後在右下、右上、左上有三個接收器，分別編號0、1、2。房間的邊長為**p**，設有一道雷射光從左下角往右上方向射出，且首次打到鏡子時距離接收器0的距離為**q**(請參考Example的圖示)，求雷射光經過多次反射後，會先擊中哪個接收器，回傳其編號。
- ```1 <= p <= 1000```
- ```0 <= q <= p```
- 題目保證雷射光最終一定會擊中接收器

### Example
![](https://i.imgur.com/m2stEeT.png)
```
Input: p = 2, q = 1
Output: 2
Explanation: The ray meets receptor 2 the first time it gets reflected back to the left wall.
```

### My Idea
想像正方形房間可以無限向上反摺堆疊，雷射光就會一直往上走，直到擊中左右兩側的某個接收器(如下圖示)。因為正方形的邊長為p，所以擊中接收器的高度一定會是**p和q的最小公倍數**。求出擊中高度後，再根據倍數判斷是擊中哪個接收器即可。
![](https://i.imgur.com/VtVKQxo.jpg =100x)


### My Code
```python=
class Solution:
    def mirrorReflection(self, p: int, q: int) -> int:
        lcm, multiple_n = self.lcm(p, q)
        p_multiple = lcm / p
        q_multiple = multiple_n
        if q_multiple % 2 == 0:
            return 2
        elif p_multiple % 2 == 0:
            return 0
        else:
            return 1

    def lcm(self, p, q):
        q_sum = q
        multiple_n = 1
        while q_sum % p != 0:
            q_sum += q
            multiple_n += 1
        return q_sum, multiple_n
```

### Submission
![](https://i.imgur.com/eukiI57.png =400x)

## (M) 394. Decode String
https://leetcode.com/problems/decode-string/

### Description
給定一個加密後的字串**s**，回傳解密完的字串。
加密格式為```k[encoded_string]```，代表encoded_string會連續重複k次，encoded_string保證不包含數字，且k保證為正整數。
- 題目字串保證為正常加密後字串，不會有多餘的空白或其他字符，必定有解
- ```1 <= s.length <= 30```
- 加密後字串中的數字，介於1~300

### Example
```
Input: s = "2[abc]3[cd]ef"
Output: "abcabccdcdcdef"
```
```
Input: s = "3[a2[c]]"
Output: "accaccacc"
```

### My Idea
此題需要使用**stack**，首先放入一個空字串，然後遍歷整個加密後字串，並根據不同情況操作stack。假設此時遍歷到的字元為c，會有以下幾種情況：
- **數字** : 更新現在的k
- **左括號**```[``` : 表示k已確定，接著會有encoded_string。所以先將k放入stack，再將一個空字串放入stack(代表初始化的encoded_string)
- **右括號**```]``` : 表示encoded_string已確定，要進行該部分的解密，然後向前合併。將stack的最後兩個元素pop取出，分別會是encoded_string與k，然後解密還原(encoded_string * k)，並接回stack最後一個元素。
- **英文字母** : 正常字元，直接接在stack的最後一個元素上。

根據以上流程遍歷完s之後，stack只會剩下一個元素，即為解密後字串。

### My Code
```python=
class Solution:
    def decodeString(self, s: str) -> str:
        stack = ['']
        k_str = ''
        for c in s:
            if c.isdigit():
                k_str += c
            elif c == '[':
                k = int(k_str)
                stack.append(k)
                stack.append('')
                k_str = ''
            elif c == ']':
                pop_str = stack.pop()
                pop_k = stack.pop()
                stack[-1] += pop_str * pop_k
            else:
                stack[-1] += c

        return stack[0]
```

### Submission
![](https://i.imgur.com/Vo5ApPO.png =400x)

## (M) 337. House Robber III
https://leetcode.com/problems/house-robber-iii/

### Description
小偷來到一個新的住宅區，該區入口只有一棟房子**root**，房子之間有路徑連接，每棟房子只會有一個進入路徑，且最多兩個離開路徑。小偷發現這住宅區的連接方式，剛好形成一個**binary tree**，且每棟房子皆有數量不等的金錢可以偷。但如果有路徑連接的兩棟房子(parent&child)都被偷，就會觸發警報。求解，在不觸發警報的情況下，最多可以偷走多少金錢。

### Example
```
Input: [3,2,3,null,3,null,1]

     3
    / \
   2   3
    \   \ 
     3   1

Output: 7 
Explanation: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.
```

### My Idea
從root開始遍歷所有房子，由於相連的房子不能都偷，所以針對眼前的房子，有以下兩種選項：
- 偷這間房子，不偷children房子
- 不偷這間房子，可偷children房子(left最大值+right最大值)

此規則可以寫成遞迴，每次回傳上述兩種選項的偷竊金額，到了root再求最大值即可。

### My Code
```python=
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rob(self, root: TreeNode) -> int:
        rob_vals = self.rob_subtree(root)
        return max(rob_vals)
        
    def rob_subtree(self, node):
        if node is None:
            return (0, 0)

        left_rob = self.rob_subtree(node.left)
        right_rob = self.rob_subtree(node.right)
        # rob this house
        val1 = node.val + left_rob[1] + right_rob[1]
        # not rob this house
        val2 = max(left_rob) + max(right_rob)

        rob_vals = (val1, val2)
        return rob_vals
```

### Submission
![](https://i.imgur.com/Xx0PCV7.png =400x)

## (H) 224. Basic Calculator
https://leetcode.com/problems/basic-calculator/

### Description
實作一個加減法計算機，給定一個簡單表達式字串**s**，回傳計算結果。
s中可能包含左括號```(```、右括號```)```、加號```+```、減號```-```、非負整數與空白。

### Example
```
Input: "(1+(4+5+2)-3)+(6+8)"
Output: 23
```

### My Idea
使用一個stack來處理括號的狀況。遍歷s，依序放入數字或符號，當遇到右括號時，就將stack中左括號之後的元素都pop出來計算，計算完的結果再放回stack。遍歷完後，將stack中剩下的元素一起計算結果，即為答案。由於非負整數可能有0，所以判斷式要注意。

### My Code
```python=
class Solution:
    def calculate(self, s: str) -> int:
        s = s.replace(' ', '')
        stack = []
        n = -1
        for c in s:
            if c.isdigit():
                if n >= 0:
                    n = n * 10 + int(c)
                else:
                    n = int(c)
            else:
                if n >= 0:
                    stack.append(n)
                    n = -1
                if c == ')':
                    element = stack.pop()
                    tmp_list = []
                    while element != '(':
                        tmp_list.append(element)
                        element = stack.pop()
                    tmp_list.reverse()
                    N = self.list_calculate(tmp_list)
                    stack.append(N)
                else:
                    stack.append(c)
        if n >= 0:
            stack.append(n)
        return self.list_calculate(stack)

    def list_calculate(self, tmp_list):
        element_n = len(tmp_list)
        n = tmp_list[0]
        i = 1
        while i < element_n:
            op = tmp_list[i]
            i += 1
            if op == '+':
                n += tmp_list[i]
            elif op == '-':
                n -= tmp_list[i]
            i += 1
        return n
```

### Submission
![](https://i.imgur.com/SpvO2bR.png =400x)

## (M) 382. Linked List Random Node
https://leetcode.com/problems/linked-list-random-node/

### Description
給定一個linked list的頭**head**，建構一個函式，每次被呼叫時，隨機回傳整個linked list中的一個value。
**Follow up** : 如果linked list超級長，長度無從得知，要如何在不用更多的空間的前提下解決此問題?

### Example
```
// Init a singly linked list [1,2,3].
ListNode head = new ListNode(1);
head.next = new ListNode(2);
head.next.next = new ListNode(3);
Solution solution = new Solution(head);

// getRandom() should return either 1, 2, or 3 randomly. Each element should have equal probability of returning.
solution.getRandom();
```

### My Idea
可以使用最直觀的做法，先遍歷整個linked list，將所有values儲存在一個list中。當函式被呼叫時，random產生一個index，取該位置的數值回傳即可。

### My Code
```python=
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:

    def __init__(self, head: ListNode):
        """
        @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node.
        """
        self.values = []
        node = head
        while node is not None:
            self.values.append(node.val)
            node = node.next
        self.val_n = len(self.values)

    def getRandom(self) -> int:
        """
        Returns a random node's value.
        """
        random_idx = random.randint(0, self.val_n - 1)
        return self.values[random_idx]


# Your Solution object will be instantiated and called as such:
# obj = Solution(head)
# param_1 = obj.getRandom()
```

### Submission
![](https://i.imgur.com/r6umpyF.png =400x)

## (M) 1306. Jump Game III
https://leetcode.com/problems/jump-game-iii/

### Description
給定一個非負整數的數組**arr**與起始index位置**start**，當你在index```i```時，可以移動到```i + arr[i]```或```i - arr[i]```的index位置。請判斷你是否可以抵達任何一個數值為0的位置，回傳True或False。
- 不可以移動超出數組的外面
- ```1 <= arr.length <= 5 * 10^4```
- ```0 <= arr[i] < arr.length```
- ```0 <= start < arr.length```

### Example
```
Input: arr = [4,2,3,0,3,1,2], start = 5
Output: true
Explanation: 
All possible ways to reach at index 3 with value 0 are: 
index 5 -> index 4 -> index 1 -> index 3 
index 5 -> index 6 -> index 4 -> index 1 -> index 3 
```

### My Idea
可以使用DFS的概念，用一個stack來記錄可以走到的位置，並且要記錄已經走過的位置，以防重複走到會造成無限迴圈。每次從stack取出一個位置，檢查其數值，如果等於0直接回傳True，如果不是0，就把向左跟向右可走到的位置放入stack。如果stack為空，但都還沒遇到0，就回傳False。

### My Code
```python=
class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        arr_n = len(arr)
        visited = set()
        stack = [start]
        while len(stack) > 0:
            idx = stack.pop()
            if (n := arr[idx]) == 0:
                return True
            visited.add(idx)
            left_idx = idx - n
            right_idx = idx + n
            if left_idx >= 0 and left_idx not in visited: 
                stack.append(left_idx)
            if right_idx < arr_n and right_idx not in visited: 
                stack.append(right_idx)
        return False
```

### Submission
![](https://i.imgur.com/AUuGs9y.png =400x)

## (E) 897. Increasing Order Search Tree
https://leetcode.com/problems/increasing-order-search-tree/

### Description
給定一個binary search tree的根節點**root**，將此樹**重新排序**，使得最左邊的node即為root，所有node都只有右節點而沒有左節點。

### Example
![](https://i.imgur.com/uPy2LoO.png =600x)
```
Input: root = [5,3,6,2,4,null,8,1,null,null,null,7,9]
Output: [1,null,2,null,3,null,4,null,5,null,6,null,7,null,8,null,9]
```

### My IDea
對binary search tree做中序遍歷即可得到由小到大的node，維護一個新的樹，將node依序向右接上即可。

### My Code
```python=
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def increasingBST(self, root: TreeNode) -> TreeNode:
        new_root = TreeNode()
        self.new_node = new_root
        self.reorder(root)
        return new_root.right

    def reorder(self, node):
        if node is not None:
            self.reorder(node.left)
            self.new_node.right = TreeNode(node.val)
            self.new_node = self.new_node.right
            self.reorder(node.right)
```

### Submission
![](https://i.imgur.com/Ut7ti6b.png =400x)

## (M) 1492. The kth Factor of n
https://leetcode.com/problems/the-kth-factor-of-n/

### Description
給定兩個正整數**n**和**k**。如果整數i滿足```n % i == 0```，則整數i為n的factor。對於n來說，會有一個由小到大排序的factor列表，回傳第k個factor的數值，如果factor數量不足k個則回傳-1。
- ```1 <= k <= n <= 1000```

### Example
```
Input: n = 12, k = 3
Output: 3
Explanation: Factors list is [1, 2, 3, 4, 6, 12], the 3rd factor is 3.
```

### My Idea
此題要求出所有n的因數，由於題目限制n在1000以下，所以可以直接從1遍歷到根號n，求所有因數，也不會很耗時。當i為n的因數時，n/i也會是n的因數，要記得算進去。

### My Code
```python=
class Solution:
    def kthFactor(self, n: int, k: int) -> int:
        factors = set([1, n])
        for i in range(2, int(n**0.5) + 1):
            if i not in factors and n % i == 0:
                factors.add(i)
                factors.add(int(n / i))
        if k > len(factors):
            return -1
        else:
            factors = list(factors)
            factors.sort()
            return factors[k-1]
```

### Submission
![](https://i.imgur.com/6rq0nUn.png =400x)

## (E) 605. Can Place Flowers
https://leetcode.com/problems/can-place-flowers/

### Description
有一個直線長形的花圃，有些位置有種花，有些還沒有，而種花的規則是：相鄰的位置不能都種花。給定一個只包含0或1的數組**flowerbed**，0表示空閒位置，1表示已經有種花。另外給定一個整數**n**，請判斷在不違反種花規則的情況下，能否多種下n個位置的花，回傳True或False。
- ```1 <= flowerbed.length <= 2 * 10^4```
- ```0 <= n <= flowerbed.length```

### Example
```
Input: flowerbed = [1,0,0,0,1], n = 1
Output: true
```
```
Input: flowerbed = [1,0,0,0,1], n = 2
Output: false
```

### My Idea
由於花圃原本就有某些位置已經種花，所以要考量的是，花與花中間的空位還能種多少花。遍歷整個flowerbed，計算連續的空位數，直到遇到1(原本已有種花)，就結算前面的連續空位能種多少花，接著重置空位計數，持續到底。要注意的是，開頭與結尾要假設都多一個空位。

### My Code
```python=
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        can_place_n = 0
        empty_count = 0
        for planted in flowerbed:
            if planted == 0:
                empty_count += 1
            else:
                can_place_n += int(empty_count / 2)
                empty_count = -1
        can_place_n += int((empty_count + 1) / 2)
        return can_place_n >= n
```

### Submission
![](https://i.imgur.com/xd3v4ZH.png =400x)

## (M) 738. Monotone Increasing Digits
https://leetcode.com/problems/monotone-increasing-digits/

### Description
給定一個非負整數**N**，回傳不大於N的最大**單調遞增數**。
- 單調遞增數：每個位數的數字一定不小於前一個位數的數字，也就是對於任意相鄰的兩個位數```xy```，一定滿足```x <= y```。
- N介於0~10^9

### Example
```
Input: N = 1234
Output: 1234
```
```
Input: N = 332
Output: 299
```

### My Idea
從第一個位數開始，向後順序檢查是否符合單調遞增，並且記錄目前有幾個連續相同的數字。如果中途發現有個位數不滿足單調遞增，就往回推到連續出現相同數字的第一個位置(如果沒有就是前一位數)，將其減1，再將後面位數全都補上9即可；如果一直到最後都滿足單調遞增，則答案就是N。

### My Code
```python=
class Solution:
    def monotoneIncreasingDigits(self, N: int) -> int:
        str_n = str(N)
        first_n = int(str_n[0])
        last_n = first_n
        ans_digits = str(first_n)
        same_count = 1
        for digit in str_n[1:]:
            n = int(digit)
            if n < last_n:
                tmp = int(ans_digits[-1]) - 1
                ans_digits = ans_digits[:-same_count] + str(tmp)
                ans_digits += '9' * (len(str_n) - len(ans_digits))
                break
            else:
                if n == last_n:
                    same_count += 1
                else:
                    same_count = 1
                ans_digits += digit
            last_n = n
        return int(ans_digits)
```

### Submission
![](https://i.imgur.com/awv5DiS.png =400x)

## (M) 54. Spiral Matrix
https://leetcode.com/problems/spiral-matrix/

### Description
給定一個```m*n```的**matrix**，回傳螺旋順序的所有元素。
- ```m == matrix.length```
- ```n == matrix[i].length```
- ```1 <= m, n <= 10```
- ```-100 <= matrix[i][j] <= 100```

### Example
![](https://i.imgur.com/AEeqTBC.png =300x)
```
Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]
```

### My Idea
螺旋狀的遍歷整個matrix，分成上下左右四種方向，針對每個方向，要紀錄一個要轉彎的idx。

### My Code
```python=
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        m = len(matrix)
        n = len(matrix[0])
        spiral = []
        i, j = 0, 0
        direction = 'right'
        complete_up_row = -1
        complete_down_row = m
        complete_left_col = -1
        complete_right_col = n
        count = 0
        total = m * n
        while count < total:
            spiral.append(matrix[i][j])
            count += 1
            if direction == 'right':
                if j + 1 == complete_right_col:
                    complete_up_row = i
                    i += 1
                    direction = 'down'
                else:
                    j += 1
            elif direction == 'down':
                if i + 1 == complete_down_row:
                    complete_right_col = j
                    j -= 1
                    direction = 'left'
                else:
                    i += 1
            elif direction == 'left':
                if j - 1 == complete_left_col:
                    complete_down_row = i
                    i -= 1
                    direction = 'up'
                else:
                    j -= 1
            elif direction == 'up':
                if i - 1 == complete_up_row:
                    complete_left_col = j
                    j += 1
                    direction = 'right'
                else:
                    i -= 1
        return spiral
```

### Submission
![](https://i.imgur.com/vLqoRZb.png =400x)

## (M) 59. Spiral Matrix II
https://leetcode.com/problems/spiral-matrix-ii/

### Description
給定一個正整數**n**，回傳一個```n*n```的matrix，其中依照螺旋順序填入1~n^2的數字。
- n介於1~20

### Example
![](https://i.imgur.com/oVWiQP9.png =250x)
```
Input: n = 3
Output: [[1,2,3],[8,9,4],[7,6,5]]
```

### My Idea
參考上面一題(54. Spiral Matrix)的解法，只是輸入輸出形式相反。由於指定了n，可以事先建出符合大小的matrix，然後依照螺旋順序由小到大在對應位置填入數字即可。

### My Code
```python=
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        matrix = [[None for i in range(n)] for i in range(n)]
        i, j = 0, 0
        direction = 'right'
        complete_up_row = -1
        complete_down_row = n
        complete_left_col = -1
        complete_right_col = n
        count = 0
        total = n * n
        while count < total:
            count += 1
            matrix[i][j] = count
            if direction == 'right':
                if j + 1 == complete_right_col:
                    complete_up_row = i
                    i += 1
                    direction = 'down'
                else:
                    j += 1
            elif direction == 'down':
                if i + 1 == complete_down_row:
                    complete_right_col = j
                    j -= 1
                    direction = 'left'
                else:
                    i += 1
            elif direction == 'left':
                if j - 1 == complete_left_col:
                    complete_down_row = i
                    i -= 1
                    direction = 'up'
                else:
                    j -= 1
            elif direction == 'up':
                if i - 1 == complete_up_row:
                    complete_left_col = j
                    j += 1
                    direction = 'right'
                else:
                    i -= 1
        return matrix
```

### Submission
![](https://i.imgur.com/EVgXajq.png =400x)

## (M) 117. Populating Next Right Pointers in Each Node II
https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/

### Description
給定一個binary tree的**root**，要讓每個節點的next指向同一層的下一個右邊節點(如果沒有就None)。
- 一開始所有node.next都是None
- 全部node數量小於6000
- ```-100 <= node.val <= 100```

### Example
![](https://i.imgur.com/zuKrnVx.png =500x)
```
Input: root = [1,2,3,4,5,null,7]
Output: [1,#,2,3,#,4,5,7,#]
Explanation: Given the above binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B. The serialized output is in level order as connected by the next pointers, with '#' signifying the end of each level.
```

### My Idea
因為是要找同一層的右邊節點，所以使用一個queue來記錄整層**由左至右**的所有非None節點(使用一個特殊符號來區隔不同層)。從queue依序取出同一層**由右至左**的節點，將next設置為上一個節點，並且將該節點的**右、左子節點依序**(不可相反)放入queue，以供下一層操作。迴圈操作直到queue中已空，代表所有節點都處理完成，回傳root即可。
因為leetcode的Python似乎無法使用Queue，所以改使用collections的deque，appendleft代替put、pop代替get。

### My Code
```python=
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if root is None:
            return root
        q = collections.deque() # queue
        q.appendleft(root)
        q.appendleft('#')
        while True:
            next_node = None
            while (node := q.pop()) is not '#':
                node.next = next_node
                next_node = node
                if (right_node := node.right) is not None:
                    q.appendleft(right_node)
                if (left_node := node.left) is not None:
                    q.appendleft(left_node)
            if q:
                q.appendleft('#')
            else:
                return root
```

### Submission
![](https://i.imgur.com/C5n6peh.png =400x)

## (M) 395. Longest Substring with At Least K Repeating Characters
https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/

### Description
給定一個字串**s**和整數**k**，求s的所有子字串中，每個子字串內字母皆出現k次以上的子字串，最長的長度。如果不存在則回傳0。
- s的長度介於1~10^4
- s中只包含小寫的英文字母
- k介於1~10^5

### Example
```
Input: s = "aaabb", k = 3
Output: 3
Explanation: The longest substring is "aaa", as 'a' is repeated 3 times.
```
```
Input: s = "ababbc", k = 2
Output: 5
Explanation: The longest substring is "ababb", as 'a' is repeated 2 times and 'b' is repeated 3 times.
```

### My Idea
我採用一層層過濾的方法。過濾函式：針對輸入字串s，遍歷統計所有字母的出現次數，保留出現次數大於等於k的字母，再遍歷一次s，找出由連續保留字母組成的子字串substr，遞迴丟入過濾函式，不斷更新最長的長度。如果s的所有字母皆保留，就回傳s的長度；如果沒有保留字母，就回傳0。

### My Code
```python=
class Solution:
    def longestSubstring(self, s: str, k: int) -> int:
        return self.str_check(s, k)
        
    def str_check(self, s, k):
        if (s_len := len(s)) < k:
            return 0
        char_count = dict()
        for i, char in enumerate(s):
            if char not in char_count:
                char_count[char] = 1
            else:
                char_count[char] += 1
        char_n = len(char_count.keys())
        keep_chars = []
        for char, count in char_count.items():
            if count >= k:
                keep_chars.append(char)
        if (l := len(keep_chars)) == 0:
            return 0 
        elif l == char_n:
            return s_len
        else:
            longest = 0
            substr = ''
            i = 0
            while i < s_len:
                if s[i] in keep_chars:
                    substr += s[i]
                elif len(substr) > 0:
                    if (max_len := self.str_check(substr, k)) > longest:
                        longest = max_len
                    substr = ''
                i += 1
            if (max_len := self.str_check(substr, k)) > longest:
                longest = max_len
            return longest
```

### Submission
![](https://i.imgur.com/cz9jPq3.png =400x)

