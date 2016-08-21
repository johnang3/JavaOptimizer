package angland.optimizer.var.matrix;

import angland.optimizer.var.scalar.Scalar;



public class ArrayMatrixValue<VarKey> extends MatrixBase<Scalar<VarKey>>
    implements
      Matrix<VarKey> {

  private final Scalar<VarKey>[] values;


  protected ArrayMatrixValue(int height, int width, Scalar<VarKey>[] values) {
    super(height, width);
    this.values = values;
  }

  @Override
  protected Scalar<VarKey>[] values() {
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

  public static <VarKey> ArrayMatrixValue<VarKey> times(Scalar<VarKey> scalar,
      Matrix<VarKey> matrix) {
    Builder<VarKey> newMatrix = new Builder<>(matrix.getHeight(), matrix.getWidth());
    for (int i = 0; i < matrix.getHeight(); ++i) {
      for (int j = 0; j < matrix.getWidth(); ++j) {
        newMatrix.set(i, j, matrix.get(i, j).times(scalar));
      }
    }
    return newMatrix.build();
  }

  public static class Builder<VarKey> extends MatrixBase<Scalar<VarKey>> {

    protected final Scalar<VarKey>[] values;

    @SuppressWarnings("unchecked")
    public Builder(int height, int width) {
      super(height, width);
      this.values = (Scalar<VarKey>[]) new Scalar[height * width];
    }

    @Override
    protected Scalar<VarKey>[] values() {
      return values;
    }

    public void set(int row, int column, Scalar<VarKey> calc) {
      validateCoords(row, column);
      values()[column + getWidth() * row] = calc;
    }

    public ArrayMatrixValue<VarKey> build() {
      return new ArrayMatrixValue<>(getHeight(), getWidth(), values);
    }

  }



}
