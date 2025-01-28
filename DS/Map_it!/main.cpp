#include "map.h"

int main() {
    Map<string,int> map1;
    map1.insert("1",1);
    map1.insert("2",2);
    map1.insert("3",3);
    map1.insert("4",4);
    map1.insert("5",6);
    map1.insert("6",6);

    cout << map1;
    cout << map1.has_duplicate_values() << endl;
    cout << map1.max_value() << endl;
    map1.trim("2","4");
    cout << map1;

    return 0;
}
