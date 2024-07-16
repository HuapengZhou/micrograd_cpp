#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <string>

namespace engine {

class Value;
using ValuePtr = std::shared_ptr<Value>;

// Value class definition
class Value : public std::enable_shared_from_this<Value> {
public:
  // Constructor
  Value(double data, std::set<ValuePtr> children = {},
        const std::string &op = "");

  // Operators
  ValuePtr operator+(const std::shared_ptr<Value> &other) const;
  ValuePtr operator+(double other) const;
  ValuePtr operator-(const std::shared_ptr<Value> &other) const;
  ValuePtr operator-(double other) const;
  ValuePtr operator*(const std::shared_ptr<Value> &other) const;
  ValuePtr operator/(const std::shared_ptr<Value> &other) const;
  ValuePtr operator/(double other) const;
  ValuePtr operator^(double other) const;
  ValuePtr ReLU() const;
  ValuePtr operator-() const;

  // Backward function for autograd
  void Backward();

  // Friend function for output stream
  friend std::ostream &operator<<(std::ostream &os, const Value &value);

private:
  double data_;
  double grad_;
  std::function<void()> backward_;
  std::set<ValuePtr> prev_;
  std::string op_;
};

// Free function for scalar + Value
ValuePtr operator+(double left, const Value &right);

} // namespace engine
