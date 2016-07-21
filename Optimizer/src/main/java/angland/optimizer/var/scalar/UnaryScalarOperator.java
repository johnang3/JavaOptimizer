package angland.optimizer.var.scalar;

import java.util.function.Consumer;

import angland.optimizer.var.ContextKey;
import angland.optimizer.var.KeyedDerivative;

public class UnaryScalarOperator<VarKey> implements IScalarValue<VarKey> {

  private final double fOfX;
  private final double fPrimeOfX;
  private final IScalarValue<VarKey> arg;

  public UnaryScalarOperator(double fOfX, double fPrimeOfX, IScalarValue<VarKey> arg) {
    super();
    this.fOfX = fOfX;
    this.fPrimeOfX = fPrimeOfX;
    this.arg = arg;
  }

  @Override
  public double value() {
    return fOfX;
  }

  @Override
  public double d(ContextKey<VarKey> key) {
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
