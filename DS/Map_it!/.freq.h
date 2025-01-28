#include ".queue_dll.h"
#include ".node.h"
#include <string>
#include <iostream>
using namespace std;


template <class T>
class FreqTable
{
public:
    // Given functions.
    FreqTable();
    FreqTable(const FreqTable& other);
    ~FreqTable() { clear(); }
    Node<T>* get_root() const { return root; }
    bool is_empty() const { return root == nullptr; }
    int freq_of(const T& val) const;
    bool remove(const T& val);
    void clear();
    DLList<T> values() const;
    FreqTable& operator=(const FreqTable& other);

    // Required Functions.
    void insert(const T& val);
    T top() const;
    int freq_of(char c) const;
    void trim(int min_freq);

private:
    Node<T>* root;
    Node<T>* top1;

    void copy_from(Node<T>* node);
    void values(DLList<T>& result, Node<T>* node) const;
    void remove_1(Node<T>* ptr, Node<T>* prev);
    void remove_2(Node<T>* node);
    void clear(Node<T>* node);
    void trim(int min_freq, Node<T>* node, Node<T>* parent);
};

template <class T>
T FreqTable<T>::top() const {
    if (is_empty()) throw string("ERROR: top() called on an empty frequency table.");
    return top1->val;
}

template <>
int FreqTable<string>::freq_of(char c) const {
    // NOTE. This can be implemented using depth-first traversal.
    //       There is no particular reason for why I chose this particular
    //       implementation.
    if (is_empty())
        return 0;

    QueueDLL<Node<string>*> queue;
    queue.enqueue(root);
    int sum = 0;

    while (!queue.is_empty()) {
        Node<string>* node = queue.dequeue();

        bool begins_with_c = node->val.length() > 0 && node->val[0] == c;
        bool go_left = node->left != nullptr && (node->val.length() > 0 && c <= node->val[0]);
        bool go_right = node->right != nullptr && (node->val.length() == 0 || c >= node->val[0]);

        //cout << node->val << ": " << go_left << " " << go_right << endl;
        if (begins_with_c)  sum += node->freq;
        if (go_left)        queue.enqueue(node->left);
        if (go_right)       queue.enqueue(node->right);
    }

    return sum;
}

template <class T>
void FreqTable<T>::trim(int min_freq, Node<T>* node, Node<T>* parent) {
    // NOTE. This can be implemented using any traversal. However, post-order is
    //       the cleanest, because if the recursive calls happen after the deletion,
    //       node might no longer exist, which requires careful handling.
    if (node == nullptr)
        return;

    trim(min_freq, node->left, node);
    trim(min_freq, node->right, node);

    if (node->freq <= min_freq) {
        if (node->left == nullptr || node->right == nullptr)
            remove_1(node, parent);
        else
            remove_2(node);
    }
}

template <class T>
void FreqTable<T>::trim(int min_freq) {
    trim(min_freq, root, nullptr);
}

template <class T>
FreqTable<T>::FreqTable() { root = nullptr; }

template <class T>
void FreqTable<T>::copy_from(Node<T>* node) 
{
    if (node == nullptr)
        return;
    
    insert(node->val);
    copy_from(node->left);
    copy_from(node->right);
}

template <class T>
FreqTable<T>::FreqTable(const FreqTable<T>& other) 
{
    root = nullptr;
    copy_from(other.root);
}

template <class T>
int FreqTable<T>::freq_of(const T& val) const
{
    Node<T>* node = root;
    
    while(node != nullptr) {
        if (val == node->val)   return node->freq;
        if (val < node->val)    node = node->left;
        else                    node = node->right;
    }

    return 0; 
}

template <class T>
void FreqTable<T>::insert(const T& val)
{
    Node<T>* curr = root;
    Node<T>* prev = nullptr;

    while (curr != nullptr) {
        prev = curr;
        if (val < curr->val)
            curr = curr->left;
        else if (val > curr->val)
            curr = curr->right;
        else {
            curr->freq++;
            if (curr->freq > top1->freq)
                top1 = curr;
            return;
        }
    }

    Node<T>* new_node = new Node<T>(val, nullptr, nullptr); 
    if (root == nullptr) {
        root = new_node;
        top1 = new_node;
    }
    else if (val < prev->val)    
        prev->left = new_node;
    else                    
        prev->right = new_node;
}

template <class T>
DLList<T> FreqTable<T>::values() const
{
    DLList<T> result;
    values(result, root);
    return result;
}

template <class T>
void FreqTable<T>::values(DLList<T>& result, Node<T>* node) const
{
    if (node == nullptr) 
        return;

    values(result, node->left);
    result.add_to_tail(node->val);
    values(result, node->right);
}


template <class T>
bool FreqTable<T>::remove(const T& val)
{
    Node<T>* node = root;
    Node<T>* prev = nullptr;

    while (node != nullptr) {
        if (node->val == val)
            break;
        
        prev = node;
        if (val < node->val)
            node = node->left;
        else
            node = node->right;
    }
    
    if (node == nullptr)
        return false;

    if (node->freq > 1)
        node->freq--;
    else if (node->left == nullptr || node->right == nullptr)
        remove_1(node, prev);
    else
        remove_2(node);
    
    return true;
}

template <class T>
void FreqTable<T>::remove_1(Node<T>* ptr, Node<T>* prev)
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

template <class T>
void FreqTable<T>::remove_2(Node<T>* node)
{
    Node<T>* rep = node->left; 
    Node<T>* prev = node;
    
    while (rep->right != nullptr) {
        prev = rep;
        rep = rep->right;
    }
    
    node->val = rep->val;
    node->freq = rep->freq;
    remove_1(rep, prev);
}


template <class T>
void FreqTable<T>::clear()
{
    clear(root);
    root = nullptr;
}
 
template <class T>
void FreqTable<T>::clear(Node<T>* node)
{
    if (node == nullptr) 
        return;

    clear(node->left);
    clear(node->right);

    delete node;
}

template <class T>
FreqTable<T>& FreqTable<T>::operator=(const FreqTable<T>& other)
{
    if (this == &other)
        return *this;
 
    clear();
    copy_from(other.root);
    return *this;
}

template <class T>
void print(ostream& out, Node<T>* node) {
    if (node == nullptr)
        return;
    print(out, node->get_left());
    out << node->get_val() << ": " << node->get_freq() << endl;
    print(out, node->get_right());
}

template <class T>
ostream& operator<<(ostream& out, const FreqTable<T>& table) {
    print(out, table.get_root());
    return out;
}


