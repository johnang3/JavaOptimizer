package angland.optimizer.var;

import java.util.Map;

public class RangeView<VarKey> implements IVectorValue<VarKey> {

  private final IMatrixValue<VarKey> source;
  private final int startRow, startCol, length, stepRow, stepCol;

  public RangeView(IMatrixValue<VarKey> source, int startRow, int startCol, int length,
      int stepRow, int stepCol) {
    super();
    this.source = source;
    this.startRow = startRow;
    this.startCol = startCol;
    this.length = length;
    this.stepRow = stepRow;
    this.stepCol = stepCol;
  }

  @Override
  public ScalarValue<VarKey> getCalculation(int row, int column) {
    if (column != 0) {
      throw new IllegalArgumentException("Vectors have only one column.");
    }
    if (row < 0) {
      throw new IllegalArgumentException("Row cannot be less than zero.");
    }
    if (row > length) {
      throw new IllegalArgumentException("Row must be less than length");
    }
    return source.getCalculation(startRow + row * stepRow, startCol + row * stepCol);
  }

  @Override
  public Map<IndexedKey<VarKey>, Double> getContext() {
    return source.getContext();
  }

  @Override
  public int getHeight() {
    return length;
  }



}
