#include "engine.h"

#include <cmath>
#include <iostream>
#include <set>
#include <functional>
#include <memory>
#include <cassert>
#include <string>
#include <vector>

namespace engine {

namespace {
using ValuePtr = std::shared_ptr<Value>;
}

// Constructor
Value::Value(double data, std::set<ValuePtr> children, const std::string& op)
    : data_(data), grad_(0), backward_([]() {}), prev_(std::move(children)), op_(op) {}

// Backward function for autograd
void Value::Backward() {
  std::set<ValuePtr> visited_set;
  std::vector<ValuePtr> reversed_value_ptrs;
  
  // Backtrack to get reversed topo flow
  std::function<void(ValuePtr)> build_topo = [&](const ValuePtr& value_ptr) {
    if (!visited_set.count(value_ptr)) {
      visited_set.insert(value_ptr);
      for (const ValuePtr& child_ptr : value_ptr->prev_) {
        build_topo(child_ptr);
      }
      reversed_value_ptrs.push_back(value_ptr);
    }
  };
  build_topo(shared_from_this());
  grad_ = 1;
  for (auto it = reversed_value_ptrs.rbegin(); it != reversed_value_ptrs.rend(); ++it) {
    (*it)->backward_();
  }
}

// Addition operator
ValuePtr Value::operator+(const std::shared_ptr<Value>& other) const {
  const ValuePtr other_value = other ? other : std::make_shared<Value>(0.0);
  ValuePtr self = std::const_pointer_cast<Value>(shared_from_this());
  ValuePtr out = std::make_shared<Value>(
      data_ + other_value->data_, std::set<ValuePtr>{self, other_value}, "+");

  auto backward = [out, self, other_value]() {
    self->grad_ += out->grad_;
    other_value->grad_ += out->grad_;
  };
  out->backward_ = backward;

  return out;
}

// Right-hand side addition operator
ValuePtr Value::operator+(double other) const {
  return *this + std::make_shared<Value>(other);
}

// Free function for scalar + Value
ValuePtr operator+(double left, const Value& right) {
  return *std::make_shared<Value>(left) + std::make_shared<Value>(right);
}

// Negation operator
ValuePtr Value::operator-() const {
  return *this * std::make_shared<Value>(-1.0);
}

// Subtraction operators
ValuePtr Value::operator-(const std::shared_ptr<Value>& other) const {
  return *this + -(*other);
}

ValuePtr Value::operator-(double other) const {
  return *this + std::make_shared<Value>(-other);
}

// Multiplication operator
ValuePtr Value::operator*(const std::shared_ptr<Value>& other) const {
  const ValuePtr other_value = other ? other : std::make_shared<Value>(0.0);
  ValuePtr self = std::const_pointer_cast<Value>(shared_from_this());
  ValuePtr out = std::make_shared<Value>(
      data_ * other_value->data_, std::set<ValuePtr>{self, other_value}, "*");

  auto backward = [out, self, other_value]() {
    self->grad_ += other_value->data_ * out->grad_;
    other_value->grad_ += self->data_ * out->grad_;
  };
  out->backward_ = backward;

  return out;
}

// Division operators
ValuePtr Value::operator/(const std::shared_ptr<Value>& other) const {
  return *this * (other->operator^(-1.0));
}

ValuePtr Value::operator/(double other) const {
  return *this * std::make_shared<Value>(1.0 / other);
}

// Power operator
ValuePtr Value::operator^(double other) const {
  assert((std::is_same<double, decltype(other)>::value) && "only supporting int/float powers for now");
  ValuePtr self = std::const_pointer_cast<Value>(shared_from_this());
  ValuePtr out = std::make_shared<Value>(std::pow(data_, other), std::set<ValuePtr>{self}, "**" + std::to_string(other));

  auto backward = [out, self, other]() {
    self->grad_ += (other * std::pow(self->data_, other - 1)) * out->grad_;
  };
  out->backward_ = backward;

  return out;
}

// ReLU activation function
ValuePtr Value::ReLU() const {
  ValuePtr self = std::const_pointer_cast<Value>(shared_from_this());
  ValuePtr out = std::make_shared<Value>(data_ < 0 ? 0 : data_, std::set<ValuePtr>{self}, "ReLU");

  auto backward = [out, self]() {
    self->grad_ += (out->data_ > 0) * out->grad_;
  };
  out->backward_ = backward;

  return out;
}

// Output stream operator
std::ostream& operator<<(std::ostream& os, const Value& value) {
  os << "Value(data=" << value.data_ << ", grad=" << value.grad_ << ")";
  return os;
}

} // namespace engine
