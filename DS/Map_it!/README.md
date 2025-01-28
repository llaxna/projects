# Overview
The **Map** class (often referred to as a "dictionary" or a "symbol table") is implemented as a binary search tree that stores in each node a key and a corresponding value. For example:
- A **Map<string, string>** object can map usernames (keys) to passwords (values), a word to its dictionary meaning, or a country to its capital city, etc.
- A **Map<string, int>** object can map student IDs (keys) to grades (values), words to the frequency of their occurrence in a book, cities to their population counts, etc.

The nodes in the BST are ordered by the key, not the value (keys can't repeat, but values can). For example, The following **Map** object maps cities to their population counts:

![image](https://github.com/user-attachments/assets/755e08e2-9f6a-4ea2-a45a-0cc9d5927f5f)

Note that **Amman > Ajloun** (because **m > j**) and **Amman < Aqaba** (because **m < q**).
