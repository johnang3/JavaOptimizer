package angland.optimizer.var;

import java.util.ArrayList;
import java.util.List;

import angland.optimizer.var.ArrayVectorValue.Builder;

public interface IVectorValue<VarKey> extends IMatrixValue<VarKey> {


  public default int getWidth() {
    return 1;
  }

  public default int getLength() {
    return getHeight();
  }

  public default ScalarValue<VarKey> getCalculation(int idx) {
    return get(idx, 0);
  }

  public default ArrayVectorValue<VarKey> plus(IMatrixValue<VarKey> other) {
    if (this.getHeight() != other.getHeight()) {
      throw new RuntimeException("Cannot add matrices of differing heights.");
    }
    if (this.getWidth() != other.getWidth()) {
      throw new RuntimeException("Cannot add matrices of differing widths");
    }
    Builder<VarKey> builder = new Builder<>(getLength());
    for (int i = 0; i < getLength(); ++i) {
      builder.set(i, this.getCalculation(i).plus(other.get(i, 0)));
    }
    return builder.build();
  }

  public default ArrayVectorValue<VarKey> softmax() {
    List<ScalarValue<VarKey>> exp = new ArrayList<>();
    for (int i = 0; i < getLength(); ++i) {
      exp.add(getCalculation(i).exp());
    }
    ScalarValue.Builder<VarKey> sumBuilder = new ScalarValue.Builder<>(exp.size());
    for (ScalarValue<VarKey> calc : exp) {
      sumBuilder.increment(calc);
    }
    ScalarValue<VarKey> sum = sumBuilder.build();
    Builder<VarKey> resultBuilder = new Builder<>(getLength());
    for (int i = 0; i < exp.size(); ++i) {
      resultBuilder.set(i, exp.get(i).divide(sum));
    }
    return resultBuilder.build();
  }

}
