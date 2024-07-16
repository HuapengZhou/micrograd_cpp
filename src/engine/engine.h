#pragma once

#include <functional>
#include <memory>
#include <set>
#include <string>

namespace engine {

class Value : public std::enable_shared_from_this<Value> {
public:
  // Constructor
  Value(double data, std::set<std::shared_ptr<Value>> children = {},
        const std::string &op = "")
      : data_(data), grad_(0), backward_([]() {}), prev_(std::move(children)),
        op_(op) {}

  // Addition operator
  std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &other) const;

  // Multiplication operator
  std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &other) const;

  // Power operator
  std::shared_ptr<Value> operator^(double other) const;

  // ReLU activation function
  std::shared_ptr<Value> ReLU() const;

  // Backward function for autograd
  void Backward();

  // Negation operator
  std::shared_ptr<Value> operator-() const;

  // Subtraction operators
  std::shared_ptr<Value> operator-(const std::shared_ptr<Value> &other) const;
  std::shared_ptr<Value> operator-(double other) const;

  // Division operators
  std::shared_ptr<Value> operator/(const std::shared_ptr<Value> &other) const;
  std::shared_ptr<Value> operator/(double other) const;

  // Print the value and its gradient
  friend std::ostream &operator<<(std::ostream &os, const Value &value);

private:
  double data_;
  double grad_;
  std::function<void()> backward_;
  std::set<std::shared_ptr<Value>> prev_;
  std::string op_;
};

} // namespace engine
