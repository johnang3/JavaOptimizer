package angland.optimizer.var.scalar;

import java.util.function.Consumer;

import angland.optimizer.var.ContextKey;
import angland.optimizer.var.KeyedDerivative;

public class StreamingProduct<VarKey> implements IScalarValue<VarKey> {

  private final IScalarValue<VarKey> left;
  private final IScalarValue<VarKey> right;
  private final double value;
  private final int branchComplexity;

  public StreamingProduct(IScalarValue<VarKey> left, IScalarValue<VarKey> right) {
    super();
    this.left = left;
    this.right = right;
    this.value = left.value() * right.value();
    this.branchComplexity = left.getBranchComplexity() + right.getBranchComplexity();
  }

  @Override
  public double value() {
    return value;
  }

  @Override
  public void actOnKeyedDerivatives(Consumer<KeyedDerivative<VarKey>> consumer) {
    if (right.value() != 0) {
      left.actOnKeyedDerivatives(kd -> consumer.accept(new KeyedDerivative<>(kd.getKey(), kd
          .getValue() * right.value())));
    }
    if (left.value() != 0) {
      right.actOnKeyedDerivatives(kd -> consumer.accept(new KeyedDerivative<>(kd.getKey(), kd
          .getValue() * left.value())));
    }
  }

  @Override
  public double d(ContextKey<VarKey> key) {
    return left.value() * right.d(key) + right.value() * left.d(key);
  }

  @Override
  public int getBranchComplexity() {
    return branchComplexity;
  }

}
