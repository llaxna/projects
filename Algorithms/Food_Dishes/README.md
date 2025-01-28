
![food](https://github.com/user-attachments/assets/bac61fdf-72b3-4b0a-816a-5ab9a5314c9b)

# Overview
A group of individuals at Princess Sumaya University for Technology were surveyed about their food preferences, after voting on a selected food dishes, it was revealed that the majority prefer dishes in the following order **(Top 10)**:

1. Mansaf(منسف)
2. Maqluba (مقلوبة)
3. Kabsa (كبسة)
4. Mosakhan (مسخن)
5. Shawarma (شاورما)
6. Fish 
7. Mlokheya (ملوخية)
8. Pizza
9. Fettuccine
10. Mandi (مندي)

Each person at PSUT can rank the list of food dishes based on how much he/she likes them. Estimating the similarity between his/her preferences and the preferences of the majority's opinion at PSUT can be measured as the distance between his/her rank and the majority’s opinion rank to the list of food. Simply, a high distance means a small value of similarity, and a low distance means a large value of similarity.

The main point here is to calculate the distance between the person’s rank and the majority's opinion rank, each dish on the person’s rank receives a penalty based on how many dishes are incorrectly ranked before.

![food1](https://github.com/user-attachments/assets/f8257bfa-9088-4705-9f0b-2b982790f734)

- Since Dr. Raghda’s ranking of the first three food dishes on majority’s opinion food list is the same, these food dishes receive no penalty.
- Majority’s opinion at PSUT ranked Fish at the 6th place in the food dishes list, but Dr. Raghda placed it before **Mosakhan** and **Shawarma** (4th and 5th rank). Therefore, it receives a penalty of **2**.
- Dr. Raghda ranked **Mosakhan** before **Shawarma**, which is fine. **Shawarma** is also not incorrectly ranked before any other book.

Therefore, the total distance is **2**.

In other words, if we consider the majority’s opinion at PSUT rankings of the food dished on Dr. Raghda's list (1, 2, 3, **6**, 4, 5), we see that the only issue is that the **6** is placed before the **4** and the **5**. Otherwise, everything is in order.

Here are more examples.

![food3](https://github.com/user-attachments/assets/a9742e56-1a61-4525-8d28-dd812da6522a)

#### 1. Distance between majority’s opinion and Dr. Ibrahim’s opinion = 0
because the rankings are identical.

#### 2. Distance between majority’s opinion and Dr. Raghda’s opinion = 2
6 is incorrectly placed before 4 and 5 (penalty = 2)

#### 3. Distance between majority’s opinion and Dr. Ebaa’s opinion = 15
6 is incorrectly placed before 1, 2, 3, 4 and 5 (penalty = 5)

5 is incorrectly placed before 1, 2, 3 and 4 (penalty = 4)

4 is incorrectly placed before 1, 2 and 3 (penalty = 3)

3 is incorrectly placed before 1 and 2 (penalty = 2)

2 is incorrectly placed before 1 (penalty = 1)

1 is not incorrectly placed before anything (penalty = 0)

Total penalty = 5 + 4 + 3 + 2 + 1

# Your Task
Write a program that reads from standard input a sequence of distinct positive integers representing your rankings of the food dishes on the food list of the majority's opinion at PSUT. Your program must output the distance between your rank and their rank. The maximum number of integers in the sequence is 10,000,000.



