package angland.optimizer.var.scalar;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.KeyedDerivative;

public class StreamingSum<VarKey> implements IScalarValue<VarKey> {

  private final List<IScalarValue<VarKey>> components;
  private final double value;
  private final int branchComplexity;

  @SafeVarargs
  public StreamingSum(IScalarValue<VarKey>... components) {
    this.components = new ArrayList<>();
    double value = 0;
    int complexity = 0;
    for (IScalarValue<VarKey> component : components) {
      this.components.add(component);
      value += component.value();
      complexity += component.getBranchComplexity();
    }
    this.value = value;
    this.branchComplexity = complexity;
  }

  public StreamingSum(List<IScalarValue<VarKey>> components) {
    this.components = components;
    double value = 0;
    int complexity = 0;
    for (IScalarValue<VarKey> component : components) {
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
    for (IScalarValue<VarKey> v : components) {
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
    for (IScalarValue<VarKey> scalar : components) {
      scalar.actOnKeyedDerivatives(consumer);
    }
  }


}
