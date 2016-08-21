package angland.optimizer.var.scalar;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.KeyedDerivative;

public class StreamingSum<VarKey> implements Scalar<VarKey> {

  private final List<Scalar<VarKey>> components;
  private final double value;
  private final int branchComplexity;

  @SafeVarargs
  public StreamingSum(Scalar<VarKey>... components) {
    this.components = new ArrayList<>();
    double value = 0;
    int complexity = 0;
    for (Scalar<VarKey> component : components) {
      this.components.add(component);
      value += component.value();
      complexity += component.getBranchComplexity();
    }
    this.value = value;
    this.branchComplexity = complexity;
    if (Double.isNaN(this.value)) {
      throw new RuntimeException("NaN value");
    }
  }

  public StreamingSum(List<Scalar<VarKey>> components) {
    this.components = components;
    double value = 0;
    int complexity = 0;
    for (Scalar<VarKey> component : components) {
      value += component.value();
      complexity += component.getBranchComplexity();
    }
    this.value = value;
    this.branchComplexity = complexity;
  }

  @Override
  public double value() {
    return value;
  }

  @Override
  public double d(IndexedKey<VarKey> key) {
    double sum = 0;
    for (Scalar<VarKey> v : components) {
      sum += v.d(key);
    }
    return sum;
  }

  @Override
  public int getBranchComplexity() {
    return branchComplexity;
  }

  @Override
  public void actOnKeyedDerivatives(Consumer<KeyedDerivative<VarKey>> consumer) {
    for (Scalar<VarKey> scalar : components) {
      scalar.actOnKeyedDerivatives(consumer);
    }
  }


}
