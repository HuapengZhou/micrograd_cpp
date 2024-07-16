#include "engine.h"

#include <iostream>

int main() {
    using namespace engine;

    ValuePtr a = std::make_shared<Value>(2.0);
    ValuePtr b = std::make_shared<Value>(3.0);
    ValuePtr c = *a + b;
    ValuePtr d = c->ReLU();

    d->Backward();

    std::cout << *a << std::endl;
    std::cout << *b << std::endl;
    std::cout << *c << std::endl;
    std::cout << *d << std::endl;

    return 0;
}
