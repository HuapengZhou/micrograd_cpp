#include "src/engine/engine.h"

namespace engine {
namespace {

using ValuePtr = std::shared_ptr<Value>;
}

// Backward function for autograd
void Value::Backward() {
  std::set<ValuePtr> visited_set;
  std::vector<ValuePtr> reversed_value_ptrs = {};
  // backtrack to get reversed topo flow
  std::function<void(ValuePtr)> build_topo = [&](const ValuePtr &value_ptr) {
    if (!visited_set.count(value_ptr)) {
      visited_set.insert(value_ptr);
      for (const ValuePtr &children_ptr : value_ptr->prev_) {
        build_topo(children_ptr);
      }
      reversed_value_ptrs.push_back(value_ptr);
    }
  };
  build_topo(shared_from_this());
  grad_ = 1;
  for (auto it = reversed_value_ptrs.rbegin(); it != reversed_value_ptrs.rend();
       ++it) {
    (*it)->backward_();
  }
}

// Addition operator
ValuePtr Value::operator+(const std::shared_ptr<Value> &other) const {
  ValuePtr other_value = other ? other : std::make_shared<Value>(0.0);
  ValuePtr out = std::make_shared<Value>(
      data_ + other_value->data_, std::set<ValuePtr>{shared_from_this(), other_value}, "+");

  auto backward = [out, this, other_value]() {
    // can change class's internal value
    const_cast<double&>(this->grad_) += out->grad_;
    const_cast<double&>(other_value->grad_) += out->grad_;
  };
  out->backward_ = backward;

  return out;
}

} // namespace engine
