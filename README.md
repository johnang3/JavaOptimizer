# JavaOptimizer

This is a simple mathematical optimizer for linear and nonlinear programs.  

###Requirements
Java 1.8

#Implementation Overview

##Scalar
The Scalar interface abstracts the value of a scalar, and its partial derivatives with respect to any variables.  It has a single type parameter, VarKey, which is used to create variables and look up partial derivatives with respect to variables. Scalars can be constructed either with the Scalar.var or Scalar.constant method.  Scalar.constant creates a new scalar object of the specified value with no partial derivatives.  Scalar.var creates a scalar object associated with the given VarKey with a derivative of 1 with respect to that varkey.  

```
Scalar<String> five = Scalar.constant(5);  
System.out.println(five.value()); // prints 5  
Scalar<String> x = Scalar.var("x", 2.0);  
System.out.println(x.value()); // prints 2  
Scalar<String> fiveX = x.times(five);  
System.out.println(fiveX.d("x")); // prints 5  
```


The Scalar interface supports several operations, which include arithmetic operations, power, exp, ln, sigmoid and tanh.  Most operators implement the .d(x) method by analyzing the derivatives of their parents.  This can become expensive when sequential operations reference the same prior step multiple times.  In these cases, the .cache method should be used to aggregate a local map of VarKeys to derivatives.

##Matrix

The Matrix interface is used to represent all vectors and matrices.  Matrices have a static height and width that are passed in when they are constructed.  The .get method may be used to access each of a Matrix's scalar elements.  Matrices can be most easily instantiated by invoking the .var method on a map of IndexedKeys to doubles.  

```
Map<IndexedKey<String>, Double> context = new HashMap<>();  
context.put(IndexedKey.matrixKey("m", 0, 0), 1.0);  
context.put(IndexedKey.matrixKey("m", 0, 1), 2.0);  
context.put(IndexedKey.matrixKey("m", 1, 0), 4.0);  
context.put(IndexedKey.matrixKey("m", 1, 1), 5.0);  
Matrix<String> matrix = Matrix.var("m", 2, 2, context);  
```

The Matrix interface supports Matrix addition and multiplication.  The .transform method may be used to return a new matrix generated by applying a unary operator to each element of the matrix.


##Optimizer

The Optimizer class has several static methods that may be used to find local minima for the value of an objective function.  The most flexible of these methods is Optimizer.optimizerWithConstraints, which may be used to minimize the value of an arbitrary function with respect to any number of arbitrary constraints.  It requires these parameters:

- getResult - Create a Result object from the given context.
- getObjective - Extract the objective value from a Result.
- zeroMinimumConstraints - A list of constraints.  Each of these functions must evaluate to zero or more for the result to be in bounds.
- penaltyTransform - A unary operator to be invoked on the weighted sum of constraint violations. 
- initialContext - The starting point.
- step - The initial step distance.
- minStep - The minimum step distance that will be bothered with.
- exceedanceTolerance - The highest amount of total constraint violation that will be tolerated.

The returned value is a Solution object, which contains both the Result object created by getResult and the variable mapping used to compute that result. 

This method is implemented by adding a penalty function of the total constraint violations to the objective function.  If the given objective function is f(x), this modified function is:

f(x) +  penaltyTransform( weight * total_constraint_violations(x))

This modified unbounded objective function is minimized with every iteration, and weight is increased.  This process continues until the sum of any constraint violations is less than exceedanceTolerance.  Here is an example of applying this method for a linear program:

```
    Map<IndexedKey<String>, Double> startingPoint = new HashMap<>();
    startingPoint.put(IndexedKey.scalarKey("x"), 0.0);
    startingPoint.put(IndexedKey.scalarKey("y"), 0.0);
    Function<Map<IndexedKey<String>, Double>, Scalar<String>> getResult = m -> {
      Scalar<String> x = Scalar.var("x", m);
      Scalar<String> y = Scalar.var("y", m);
      return x.plus(y).times(Scalar.constant(-1));
    };
    List<Function<Map<IndexedKey<String>, Double>, Scalar<String>>> zeroMinimumConstraints =
        new ArrayList<>();
    zeroMinimumConstraints.add(m -> Scalar.var("x", m));
    zeroMinimumConstraints.add(m -> Scalar.var("y", m));
    zeroMinimumConstraints.add(m -> {
      Scalar<String> x = Scalar.var("x", m);
      Scalar<String> y = Scalar.var("y", m);
      return x.plus(y.times(Scalar.constant(.5))).minus(Scalar.constant(3))
          .times(Scalar.constant(-1));
    });
    zeroMinimumConstraints.add(m -> {
      Scalar<String> x = Scalar.var("x", m);
      Scalar<String> y = Scalar.var("y", m);
      return x.times(Scalar.constant(.5)).plus(y).minus(Scalar.constant(3))
          .times(Scalar.constant(-1));
    });

    Solution<Scalar<String>, String> result =
        Optimizer.optimizeWithConstraints(getResult, x -> x, zeroMinimumConstraints, Scalar::exp,
            startingPoint, 1.0, 10E-8, 10E-8);
    System.out.println(result.getContext().get(IndexedKey.scalarKey("x")));  //prints 1.9999999657714576 for x
    System.out.println(result.getContext().get(IndexedKey.scalarKey("y")));  //prints 1.9999999657714576 for y
```

And a nonlinear program:

```
    Map<IndexedKey<String>, Double> startingPoint = new HashMap<>();
    startingPoint.put(IndexedKey.scalarKey("x"), 0.0);
    startingPoint.put(IndexedKey.scalarKey("y"), 0.0);
    Function<Map<IndexedKey<String>, Double>, Scalar<String>> getResult = m -> {
      Scalar<String> x = Scalar.var("x", m);
      Scalar<String> y = Scalar.var("y", m);
      return x.power(2.0).times(y.power(3.0)).times(Scalar.constant(-1));
    };
    List<Function<Map<IndexedKey<String>, Double>, Scalar<String>>> zeroMinimumConstraints =
        new ArrayList<>();
    zeroMinimumConstraints.add(m -> Scalar.var("x", m));
    zeroMinimumConstraints.add(m -> Scalar.var("y", m));
    zeroMinimumConstraints.add(m -> {
      Scalar<String> x = Scalar.var("x", m);
      Scalar<String> y = Scalar.var("y", m);
      return x.plus(y).minus(Scalar.constant(10)).times(Scalar.constant(-1));
    });
    Solution<Scalar<String>, String> result =
        Optimizer.optimizeWithConstraints(getResult, x -> x, zeroMinimumConstraints, Scalar::exp,
            startingPoint, 1.0, 10E-8, 10E-8);
    System.out.println(result.getContext().get(IndexedKey.scalarKey("x"))); //prints 4.000017060503225 for x
    System.out.println(result.getContext().get(IndexedKey.scalarKey("y"))); //prints 5.999982806293364 for y

```



