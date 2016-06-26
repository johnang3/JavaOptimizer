package angland.optimizer.var;

import java.util.Map;

import angland.optimizer.var.ArrayMatrixValue.Builder;

public interface IMatrixValue<VarKey> {

  public int getHeight();

  public int getWidth();

  public ScalarValue<VarKey> getCalculation(int row, int column);

  public Map<IndexedKey<VarKey>, Double> getContext();

  public default IMatrixValue<VarKey> add(IMatrixValue<VarKey> other) {
    if (this.getHeight() != other.getHeight()) {
      throw new RuntimeException("Cannot add matrices of differing heights.");
    }
    if (this.getWidth() != other.getWidth()) {
      throw new RuntimeException("Cannot add matrices of differing widths");
    }
    Builder<VarKey> newMatrix = new Builder<>(getHeight(), getWidth());
    for (int i = 0; i < getHeight(); ++i) {
      for (int j = 0; j < getHeight(); ++j) {
        newMatrix.set(i, j, this.getCalculation(i, j).plus(other.getCalculation(i, j)));
      }
    }
    return newMatrix.build(other.getContext());
  }

  public default IMatrixValue<VarKey> times(IMatrixValue<VarKey> other) {
    if (this.getWidth() != other.getHeight()) {
      throw new IllegalArgumentException("Width of left matrix (" + getWidth()
          + ") must equal height of right matrix (" + other.getHeight() + ")");
    }
    Builder<VarKey> builder = new Builder<>(this.getHeight(), other.getWidth());
    for (int i = 0; i < this.getHeight(); ++i) {
      for (int j = 0; j < other.getWidth(); ++j) {
        ScalarValue.Builder<VarKey> sumBuilder = new ScalarValue.Builder<>(this.getWidth() * 3);
        for (int k = 0; k < this.getWidth(); ++k) {
          ScalarValue<VarKey> left = this.getCalculation(i, k);
          ScalarValue<VarKey> right = other.getCalculation(k, j);
          sumBuilder.incrementValue(left.value() * right.value());
          for (Map.Entry<IndexedKey<VarKey>, Double> entry : left.getGradient().entrySet()) {
            if (entry.getValue() != 0) {
              sumBuilder.getGradient().merge(entry.getKey(), entry.getValue() * right.value(),
                  Double::sum);
            }
          }
          for (Map.Entry<IndexedKey<VarKey>, Double> entry : right.getGradient().entrySet()) {
            if (entry.getValue() != 0) {
              sumBuilder.getGradient().merge(entry.getKey(), entry.getValue() * left.value(),
                  Double::sum);
            }
          }
        }
        builder.set(i, j, sumBuilder.build(getContext()));
      }
    }
    return builder.build(getContext());
  }

  public default IMatrixValue<VarKey> transpose() {
    return new TransposeView<>(this);
  }

}
