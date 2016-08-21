package angland.optimizer.var.scalar;

import java.util.function.Consumer;

import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.KeyedDerivative;

public class UnaryScalarOperator<VarKey> implements Scalar<VarKey> {

  private final double fOfX;
  private final double fPrimeOfX;
  private final Scalar<VarKey> arg;

  public UnaryScalarOperator(double fOfX, double fPrimeOfX, Scalar<VarKey> arg) {
    super();
    this.fOfX = fOfX;
    this.fPrimeOfX = fPrimeOfX;
    this.arg = arg;
    if (Double.isNaN(fOfX)) {
      throw new RuntimeException("NaN value");
    }
  }

  @Override
  public double value() {
    return fOfX;
  }

  @Override
  public double d(IndexedKey<VarKey> key) {
    return fPrimeOfX == 0 ? 0 : fPrimeOfX * arg.d(key);
  }


  @Override
  public int getBranchComplexity() {
    return arg.getBranchComplexity();
  }

  @Override
  public void actOnKeyedDerivatives(Consumer<KeyedDerivative<VarKey>> consumer) {
    if (fPrimeOfX != 0) {
      arg.actOnKeyedDerivatives(kd -> consumer.accept(new KeyedDerivative<>(kd.getKey(), kd
          .getValue() * fPrimeOfX)));
    }
  }

}
