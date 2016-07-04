package angland.optimizer.var;


public class MatrixRangeView<VarKey> implements IMatrixValue<VarKey> {

  private final IMatrixValue<VarKey> source;
  private final int startRow;
  private final int startCol;
  private final int height;
  private final int width;

  public MatrixRangeView(IMatrixValue<VarKey> source, int startRow, int startCol, int height,
      int width) {
    super();
    this.source = source;
    this.startRow = startRow;
    this.startCol = startCol;
    this.height = height;
    this.width = width;
  }

  @Override
  public int getHeight() {
    return height;
  }

  @Override
  public int getWidth() {
    return width;
  }

  @Override
  public ScalarValue<VarKey> get(int row, int column) {
    return source.get(startRow + row, startCol + column);
  }



}
