package angland.optimizer.var;

import java.util.Map;


public class ArrayMatrixValue<VarKey> extends MatrixBase<ScalarValue<VarKey>>
    implements
      IMatrixValue<VarKey> {

  private final ScalarValue<VarKey>[] values;


  protected ArrayMatrixValue(int height, int width, ScalarValue<VarKey>[] values) {
    super(height, width);
    this.values = values;
  }

  @Override
  protected ScalarValue<VarKey>[] values() {
    return values;
  }

  public ArrayMatrixValue<VarKey> add(ArrayMatrixValue<VarKey> other) {
    if (this.getHeight() != other.getHeight()) {
      throw new RuntimeException("Cannot add matrices of differing heights.");
    }
    if (this.getWidth() != other.getWidth()) {
      throw new RuntimeException("Cannot add matrices of differing widths");
    }
    Builder<VarKey> newMatrix = new Builder<>(getHeight(), getWidth());
    for (int i = 0; i < getHeight(); ++i) {
      for (int j = 0; j < getHeight(); ++j) {
        newMatrix.set(i, j, this.get(i, j).plus(other.get(i, j)));
      }
    }
    return newMatrix.build();
  }

  public static <VarKey> ArrayMatrixValue<VarKey> times(ScalarValue<VarKey> scalar,
      IMatrixValue<VarKey> matrix) {
    Builder<VarKey> newMatrix = new Builder<>(matrix.getHeight(), matrix.getWidth());
    for (int i = 0; i < matrix.getHeight(); ++i) {
      for (int j = 0; j < matrix.getWidth(); ++j) {
        newMatrix.set(i, j, matrix.get(i, j).times(scalar));
      }
    }
    return newMatrix.build();
  }

  public static class Builder<VarKey> extends MatrixBase<ScalarValue<VarKey>> {

    protected final ScalarValue<VarKey>[] values;

    @SuppressWarnings("unchecked")
    public Builder(int height, int width) {
      super(height, width);
      this.values = (ScalarValue<VarKey>[]) new ScalarValue[height * width];
    }

    @Override
    protected ScalarValue<VarKey>[] values() {
      return values;
    }

    public void set(int row, int column, ScalarValue<VarKey> calc) {
      validateCoords(row, column);
      values()[column + getWidth() * row] = calc;
    }

    public ArrayMatrixValue<VarKey> build() {
      return new ArrayMatrixValue<>(getHeight(), getWidth(), values);
    }

  }

  @Override
  public Map<IndexedKey<VarKey>, Double> getContext() {
    // TODO Auto-generated method stub
    return null;
  }



}
