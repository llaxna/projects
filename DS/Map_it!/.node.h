template <class T>
class FreqTable;

template <class T>
class Node
{
public:
    Node(const T& val, Node* left, Node* right);

    T get_val() const { return val; }
    int get_freq() const { return freq; }

    Node* get_left() const { return left; }
    Node* get_right() const { return right; }

private:
    T val;
    int freq;

    Node* left;
    Node* right;

    friend class FreqTable<T>;
};

template <class T>
Node<T>::Node(const T& val, Node* left, Node* right)
{
    this->val = val;
    this->freq = 1;

    this->left = left;
    this->right = right;
}
