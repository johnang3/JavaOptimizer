# JavaOptimizer
Java optimizer

This is a simple mathematical optimizer for linear and nonlinear programs.  

#Requirements
Java 1.8

#Overview

##Scalar
The Scalar class encapsulates the value of a scalar, and its partial derivatives with respect to any variables.  It has a single type parameter, VarKey, which is used to create variables and look up partial derivatives with respect to variables. Scalars can be constructed either with the Scalar.var or Scalar.constant method.  Scalar.constant creates a new scalar object of the specified value with no partial derivatives.  Scalar.var creates a scalar object associated with the given VarKey with a derivative of 1 with respect to that varkey and a value equal to the value of the specified varkey in the given context.  

`
Scalar<String> five = Scalar.constant(5);
System.out.println(five.value());  //prints 5
Scalar<String> x = Scalar.constant(5,"x");
`


Scalar supports several operations, which currently include arithmetic operations, exp, ln, sigmoid and tanh.  Most operations support the 




