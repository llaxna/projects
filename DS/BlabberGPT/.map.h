#include "queue_dll.h"
#include <iomanip>
#include <iostream>
using namespace std;

template <class K, class V>
class Node {
public:
    K key;
    V value;

    Node* left;
    Node* right;

    Node(const K& key, const V& value, Node* left, Node* right) {
        this->key = key;
        this->value = value;
        this->left = left;
        this->right = right;
    }
};

template <class K, class V>
class Map {
    // You are allowed to MODIFY the constructor.
    // You are NOT allowed to modify any other given function.
    // You are NOT allowed to ADD new public functions.
    
public:
    // ---------------- GIVEN FUNCTIONS ------------------- //
    Map();
    Map(const Map& other);
    ~Map() { clear(root); }

    bool        is_empty() const  { return root == nullptr; }
    bool        contains(const K&);
    V&          value_of(const K&);
    void        insert(const K&, const V&);
    void        clear();
    bool        remove(const K&);
    DLList<K>   keys() const;
    Node<K, V>* get_root() { return root; }
    Map&        operator=(const Map&);
    Map&        operator=(Map&&);

    template <class Key, class Val> friend
    ostream& operator<<(ostream& out, const Map<Key, Val>& dict);
    // ----------------------------------------------------//


    // ---------IMPLEMENT THE FOLLOWING FUNCTIONS ---------//
    Map  cut(const K& key);
    void trim(const K& lo, const K& hi);
    bool    has_duplicate_values() const;
    V    max_value() const;

private:
    Node<K, V>* search(const K& key);
    void        trim(const K& lo, const K& hi, Node<K, V>*, Node<K, V>*);
    // ----------------------------------------------------//

private:
    // You are allowed to ADD private data members
    Node<K, V>* root;
    Node<K, V>* latest;

    // You are allowed to ADD private functions
    Node<K, V>* copy_from(Node<K, V>* node);
    void clear(Node<K, V>* node);
    void keys(DLList<K>& result, Node<K, V>* node) const;
    void remove_1(Node<K, V>* ptr, Node<K, V>* prev);
    void remove_2(Node<K, V>* ptr);
};


// ------- REQUIRED FUNCTIONS --------- //

// This function returns a pointer to the node containing 
// the given key. If the key is not in the tree, the function 
// returns nullptr. The function must run in O(height) if the
// previous call to the function was for a different key. If
// the previous call was for the same key, the function must 
// run in O(1).
template <class K, class V>
Node<K, V>* Map<K, V>::search(const K& key) 
{
    if (latest != nullptr && latest->key == key)
        return latest;

    Node<K, V>* curr = root;
    while (curr != nullptr) {
        if (key == curr->key) {
            latest = curr;
            return curr;
        }
        else if (key  < curr->key) 
            curr = curr->left;
        else                       
            curr = curr->right; 
    }

    return nullptr;
}

// This function removes the subtree rooted at key and returns 
// a copy of it. It returns an empty Map object if key is not 
// in the tree. The function must run in O(height).
template <class K, class V>
Map<K, V> Map<K, V>::cut(const K& key) 
{ 
    Map<K, V> result;
    
    Node<K, V>* curr = root;
    Node<K, V>* prev = nullptr;
    while (curr != nullptr) {
        if (key == curr->key) 
            break;
        prev = curr;
        if (key  < curr->key) curr = curr->left;
        else                  curr = curr->right; 
    }

    if (curr != nullptr) {
        latest = nullptr;
        if (prev == nullptr) root = nullptr;
        else if (key < prev->key) prev->left = nullptr;
        else                 prev->right = nullptr;
        result.root = curr;
    }

    return result;
}

// This function removes from the tree all the keys larger than
// hi and all the keys less than lo. The function must run in O(n).
template <class K, class V>
void Map<K, V>::trim(const K& lo, const K& hi)
{
    if (latest != nullptr) {
        if (latest->key < lo || latest->key > hi)
         latest = nullptr;
    }
    trim(lo, hi, root, nullptr);
}

template <class K, class V>
void Map<K, V>::trim(const K& lo, const K& hi, Node<K, V>* node, Node<K, V>* prev)
{
    if (node == nullptr)
        return;

    trim(lo, hi, node->left, node);
    trim(lo, hi, node->right, node);

    if (node->key < lo || node->key > hi)
        remove_1(node, prev);
}

// This function returns the maximum value (not key) that is in
// the tree. The function:
//     - throws a string exception if the tree is empty.
//     - must run in O(n).
// Returns true if there are duplicate values (not keys)
// in the tree, and false otherwise.
// The function must run in O(n x height)
template <class K, class V>
bool Map<K, V>::has_duplicate_values() const 
{
    if (is_empty())
        return false;

    QueueDLL<Node<K, V>*> q;
    q.enqueue(root);
    Map<V, V> temp;

    while (!q.is_empty()) {
        Node<K, V>* node = q.dequeue();
        if (temp.contains(node->value))
            return true;
        temp.insert(node->value, node->value);

        if (node->left) q.enqueue(node->left);
        if (node->right) q.enqueue(node->right);
    }

    return false;
}

// This function returns the maximum value (not key) that is in
// the tree. The function:
//     - throws a string exception if the tree is empty.
//     - must run in O(n).
template <class K, class V>
V Map<K, V>::max_value() const 
{
    if (root == nullptr)
        throw string("No max value in empty tree.");

    DLList<Node<K, V>*> queue;
    queue.add_to_head(root);
    Node<K, V>* max_node = nullptr;

    while (!queue.is_empty()) {
        Node<K, V>* node = queue.tail_val();
        queue.remove_tail();

        if (!max_node || max_node->value < node->value)
            max_node = node;
        
        if (node->left) queue.add_to_head(node->left);
        if (node->right) queue.add_to_head(node->right);
    }

    return max_node->value;
}
// ------------------------------------------ //


// ------------ GIVEN FUNCTIONS ------------- //
template <class K, class V>
Map<K, V>::Map() { 
    latest = nullptr;
    root = nullptr; 
}

template <class K, class V>
void Map<K, V>::insert(const K& key, const V& value)
{
    Node<K, V>* curr = root;
    Node<K, V>* prev = nullptr;

    while (curr != nullptr) {
        if (curr->key == key) {
            curr->value = value;
            return;
        }

        prev = curr;
        if (key < curr->key) curr = curr->left;
        else                 curr = curr->right; 
    }

    Node<K, V>* new_node = new Node<K, V>(key, value, nullptr, nullptr);
    if (root == nullptr)      root = new_node;
    else if (key < prev->key) prev->left = new_node;
    else                      prev->right = new_node; 
}

template <class K, class V>
Node<K, V>* Map<K, V>::copy_from(Node<K, V>* node) 
{
    if (node == nullptr) 
        return nullptr;

    Node<K, V>* new_node = new Node<K, V>(node->key, node->value, nullptr, nullptr);
    new_node->left = copy_from(node->left);
    new_node->right = copy_from(node->right);

    return new_node;
}

template <class K, class V>
Map<K, V>::Map(const Map<K, V>& other) 
{
    latest = nullptr;
    root = copy_from(other.root);
}


template <class K, class V>
bool Map<K, V>::contains(const K& key)
{
    return search(key) != nullptr;
}

template <class K, class V>
V& Map<K, V>::value_of(const K& key)
{
    Node<K, V>* node = search(key);
    if (node == nullptr)
        throw string("No value for the given key");
    return node->value;
}

template <class K, class V>
DLList<K> Map<K, V>::keys() const
{
    DLList<K> result;
    keys(result, root);
    return result;
}

template <class K, class V>
void Map<K, V>::keys(DLList<K>& result, Node<K, V>* node) const
{
    if (node == nullptr) 
        return;

    keys(result, node->left);
    result.add_to_tail(node->key);
    keys(result, node->right);
}
 
template <class K, class V>
void Map<K, V>::clear(Node<K, V>* node)
{
    if (node == nullptr) 
        return;

    clear(node->left);
    clear(node->right);

    delete node;
}

template <class K, class V>
Map<K, V>& Map<K, V>::operator=(const Map<K, V>& other)
{
    if (this == &other)
        return *this;
 
    clear(root);
    root = copy_from(other.root);
    return *this;
}

template <class K, class V>
Map<K, V>& Map<K, V>::operator=(Map<K, V>&& other)
{
    if (this == &other)
        return *this;
 
    clear(root);
    root = other.root;
    other.root = nullptr;
    return *this;
}

template <class K, class V>
void print(ostream& out, Node<K, V>* node) {
    if (node == nullptr)
        return;
    print(out, node->left);
    out << left << node->key << ": " << node->value << endl;
    print(out, node->right);
}

template <class K, class V>
void Map<K, V>::clear() {
    clear(root);
    root = nullptr;
}

template <class K, class V>
bool Map<K, V>::remove(const K& key)
{
    Node<K, V>* node = root;
    Node<K, V>* prev = nullptr;

    while (node != nullptr) {
        if (node->key == key)
            break;
        
        prev = node;
        if (key < node->key) node = node->left;
        else                 node = node->right;
    }
    
    if (node == nullptr)
        return false;
    
    if (node->left == nullptr || node->right == nullptr)
        remove_1(node, prev);
    else
        remove_2(node);
    
    return true;
}

template <class K, class V>
void Map<K, V>::remove_2(Node<K, V>* node)
{
    Node<K, V>* rep = node->left; 
    Node<K, V>* prev = node;
    
    while (rep->right != nullptr) {
        prev = rep;
        rep = rep->right;
    }
    
    node->value = rep->value;
    remove_1(rep, prev);
}

template <class K, class V>
void Map<K, V>::remove_1(Node<K, V>* ptr, Node<K, V>* prev)
{
    if (ptr == root) {
        if (root->left != nullptr)
            root = root->left;
        else
            root = root->right;
    } 
    else if (ptr == prev->left) {
        if (ptr->right != nullptr)
            prev->left = ptr->right;
        else
            prev->left = ptr->left;
    }  
    else {
        if (ptr->right != nullptr)
            prev->right = ptr->right;
        else
            prev->right = ptr->left;
    }
    
    delete ptr;
}

template <class K, class V>
ostream& operator<<(ostream& out, const Map<K, V>& dict) {
    print(out, dict.root);
    return out;
}
