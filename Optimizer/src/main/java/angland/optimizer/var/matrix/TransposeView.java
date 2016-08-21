package angland.optimizer.var.matrix;

import angland.optimizer.var.scalar.Scalar;


public class TransposeView<VarKey> implements Matrix<VarKey> {

  private final Matrix<VarKey> source;

  public TransposeView(Matrix<VarKey> source) {
    super();
    this.source = source;
  }

  @Override
  public int getHeight() {
    return source.getWidth();
  }

  @Override
  public int getWidth() {
    return source.getHeight();
  }

  @Override
  public Scalar<VarKey> get(int row, int column) {
    return source.get(column, row);
  }


  @Override
  public Matrix<VarKey> transpose() {
    return source;
  }

}
