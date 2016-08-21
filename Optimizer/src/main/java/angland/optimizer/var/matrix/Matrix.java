package angland.optimizer.var.matrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.BinaryOperator;
import java.util.function.Function;

import angland.optimizer.var.IndexedKey;
import angland.optimizer.var.matrix.ArrayMatrixValue.Builder;
import angland.optimizer.var.scalar.MappedDerivativeScalar;
import angland.optimizer.var.scalar.Scalar;
import angland.optimizer.var.scalar.StreamingSum;

/**
 * Encapsulates a matrix of Scalars.
 * 
 * @author John Angland
 *
 * @param <VarKey>
 */
public interface Matrix<VarKey> {

  public int getHeight();

  public int getWidth();

  public Scalar<VarKey> get(int row, int column);

  public static <VarKey> Matrix<VarKey> var(VarKey key, int height, int width,
      Map<IndexedKey<VarKey>, Double> context) {
    ArrayMatrixValue.Builder<VarKey> builder = new ArrayMatrixValue.Builder<VarKey>(height, width);
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        IndexedKey<VarKey> indexedKey = IndexedKey.matrixKey(key, i, j);
        if (indexedKey == null) {
          throw new RuntimeException("Null context key for indexed key " + indexedKey);
        }
        builder.set(i, j, Scalar.var(indexedKey, context));
      }
    }
    return builder.build();
  }

  public static <VarKey> Matrix<VarKey> varOrConst(VarKey key, int height, int width,
      Map<IndexedKey<VarKey>, Double> context, boolean constant) {
    Matrix<VarKey> var = var(key, height, width, context);
    return constant ? var.toConstant() : var;
  }

  public static <VarKey> Matrix<VarKey> repeat(Scalar<VarKey> val, int height, int width) {
    ArrayMatrixValue.Builder<VarKey> builder = new ArrayMatrixValue.Builder<VarKey>(height, width);
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        builder.set(i, j, val);
      }
    }
    return builder.build();
  }

  public default Scalar<VarKey> elementSumStream() {
    List<Scalar<VarKey>> components = new ArrayList<>();
    for (int i = 0; i < getHeight(); ++i) {
      for (int j = 0; j < getWidth(); ++j) {
        components.add(get(i, j));
      }
    }
    return new StreamingSum<>(components);
  }

  /**
   * Vertically concatenates this matrix and another matrix.
   * 
   * @param other
   * @return
   */
  public default Matrix<VarKey> vCat(Matrix<VarKey> other) {
    if (getWidth() != other.getWidth()) {
      throw new IllegalArgumentException("Can only vCat matrices with the same width.");
    }
    ArrayMatrixValue.Builder<VarKey> builder =
        new ArrayMatrixValue.Builder<>(getHeight() + other.getHeight(), other.getWidth());
    for (int i = 0; i < getHeight(); ++i) {
      for (int j = 0; j < getWidth(); ++j) {
        builder.set(i, j, get(i, j));
      }
    }
    for (int i = 0; i < other.getHeight(); ++i) {
      for (int j = 0; j < other.getWidth(); ++j) {
        builder.set(i + getHeight(), j, other.get(i, j));
      }
    }
    return builder.build();
  }


  public default Matrix<VarKey> pointwise(Matrix<VarKey> other, BinaryOperator<Scalar<VarKey>> op) {
    if (this.getHeight() != other.getHeight()) {
      throw new RuntimeException(
          "Cannot perform pointwise operations matrices of differing heights.");
    }
    if (this.getWidth() != other.getWidth()) {
      throw new RuntimeException("Cannot perform pointwise operations matrices of differing widths");
    }
    Builder<VarKey> newMatrix = new Builder<>(getHeight(), getWidth());
    for (int i = 0; i < getHeight(); ++i) {
      for (int j = 0; j < getWidth(); ++j) {
        newMatrix.set(i, j, op.apply(this.get(i, j), other.get(i, j)));
      }
    }
    return newMatrix.build();
  }

  public default Matrix<VarKey> plus(Matrix<VarKey> other) {
    return pointwise(other, Scalar::plus);
  }


  public default Matrix<VarKey> pointwiseMultiply(Matrix<VarKey> other) {
    return pointwise(other, Scalar::times);
  }

  public default Matrix<VarKey> columnProximity(Matrix<VarKey> other) {
    List<Integer> list = new ArrayList<>();
    for (int i = 0; i < getWidth(); ++i) {
      list.add(i);
    }
    return columnProximity(other, list);
  }

  public default Matrix<VarKey> columnProximity(Matrix<VarKey> other, List<Integer> selectedIndices) {
    if (this.getHeight() != other.getHeight()) {
      throw new IllegalArgumentException("Cannot compare columns in matrices of differing heights");
    }
    if (other.getWidth() != 1) {
      throw new IllegalArgumentException(
          "Can only compare column proximity to a matrix of width 1.");
    }
    Builder<VarKey> newMatrix = new Builder<>(1, getWidth());
    for (int i = 0; i < getWidth(); ++i) {
      newMatrix.set(0, i, Scalar.constant(Double.MAX_VALUE));
    }
    for (int i : selectedIndices) {
      List<Scalar<VarKey>> components = new ArrayList<>(getWidth());
      for (int j = 0; j < getHeight(); ++j) {
        components.add(this.get(j, i).minus(other.get(j, 0)).power(2));
      }
      newMatrix.set(0, i, new StreamingSum<>(components).power(.5).cache());
    }
    return newMatrix.build();
  }

  public default Matrix<VarKey> sampledColumnProximity(Matrix<VarKey> other, int requiredIndex,
      int samples) {
    if (samples > getWidth()) {
      throw new IllegalArgumentException("Samples out of range: " + samples);
    }
    List<Integer> available = new ArrayList<>();
    for (int i = 0; i < getWidth(); ++i) {
      if (i != requiredIndex) {
        available.add(i);
      }
    }
    Collections.shuffle(available);
    List<Integer> selected = new ArrayList<>(samples + 1);
    for (int i = 0; i < samples; ++i) {
      selected.add(available.get(i));
    }
    selected.add(requiredIndex);
    Matrix<VarKey> m = columnProximity(other, selected);
    return m;
  }

  public default Matrix<VarKey> times(Matrix<VarKey> other) {
    if (this.getWidth() != other.getHeight()) {
      throw new IllegalArgumentException("Width of left matrix (" + getWidth()
          + ") must equal height of right " + "matrix (" + other.getHeight() + ")");
    }
    Builder<VarKey> builder = new Builder<>(this.getHeight(), other.getWidth());
    for (int i = 0; i < this.getHeight(); ++i) {
      for (int j = 0; j < other.getWidth(); ++j) {
        MappedDerivativeScalar.Builder<VarKey> sumBuilder =
            new MappedDerivativeScalar.Builder<>(this.getWidth() * 3);
        for (int k = 0; k < this.getWidth(); ++k) {
          Scalar<VarKey> left = this.get(i, k);
          Scalar<VarKey> right = other.get(k, j);
          sumBuilder.incrementValue(left.value() * right.value());

          for (Map.Entry<IndexedKey<VarKey>, Double> entry : left.getGradient().entrySet()) {
            if (entry.getValue() != 0) {
              sumBuilder.getGradient().merge(entry.getKey(), entry.getValue() * right.value());

            }
          }
          for (Map.Entry<IndexedKey<VarKey>, Double> entry : right.getGradient().entrySet()) {
            if (entry.getValue() != 0) {
              sumBuilder.getGradient().merge(entry.getKey(), entry.getValue() * left.value());
            }
          }
        }
        builder.set(i, j, sumBuilder.build());
      }
    }
    return builder.build();
  }

  /**
   * Identical to times, except that it does not cache the result.
   * 
   * @param other
   * @return
   */
  public default Matrix<VarKey> streamingTimes(Matrix<VarKey> other) {
    if (this.getWidth() != other.getHeight()) {
      throw new IllegalArgumentException("Width of left matrix (" + getWidth()
          + ") must equal height of right " + "matrix (" + other.getHeight() + ")");
    }
    Builder<VarKey> builder = new Builder<>(this.getHeight(), other.getWidth());
    for (int i = 0; i < this.getHeight(); ++i) {
      for (int j = 0; j < other.getWidth(); ++j) {
        List<Scalar<VarKey>> sumComponents = new ArrayList<>(2 * this.getWidth());
        for (int k = 0; k < this.getWidth(); ++k) {
          Scalar<VarKey> left = this.get(i, k);
          Scalar<VarKey> right = other.get(k, j);
          sumComponents.add(left.times(right));
        }
        builder.set(i, j, new StreamingSum<>(sumComponents));
      }
    }
    return builder.build();
  }

  public default Matrix<VarKey> transpose() {
    return new TransposeView<>(this);
  }

  /**
   * 
   * Return a new matrix where all values are rows are multiplied by zero except for the provided
   * index, and an additional provided number of randomly selected indices.
   * 
   * @return
   */
  public default ArrayMatrixValue<VarKey> selectAndSampleRows(int forcedIndex, int sampleCount) {
    if (forcedIndex < 0 || forcedIndex >= getHeight()) {
      throw new IllegalArgumentException("Forced index out of range: " + forcedIndex);
    }
    if (sampleCount < 0) {
      throw new IllegalArgumentException("Sample count cannot be negative");
    }
    if (sampleCount > getHeight() - 1) {
      throw new IllegalArgumentException("Sample count cannot be greater than height-1");
    }
    ArrayMatrixValue.Builder<VarKey> builder = new ArrayMatrixValue.Builder<>(getHeight(), 1);
    Scalar<VarKey> zero = Scalar.constant(0);
    for (int i = 0; i < getHeight(); ++i) {
      for (int j = 0; j < getWidth(); ++j) {
        builder.set(i, j, zero);
      }
    }
    for (int i = 0; i < getWidth(); ++i) {
      builder.set(forcedIndex, i, get(forcedIndex, i));
    }
    List<Integer> availableIndices = new ArrayList<>(this.getHeight() - 1);
    for (int i = 0; i < getHeight(); ++i) {
      if (i != forcedIndex) {
        availableIndices.add(i);
      }
    }
    Collections.shuffle(availableIndices);
    for (int i = 0; i < sampleCount; ++i) {
      int row = availableIndices.get(i);
      for (int j = 0; j < getWidth(); ++j) {
        builder.set(row, j, get(row, j));
      }
    }
    return builder.build();
  }

  /**
   * Returns a list of integers containing the provided forced parameter, followed by an additional
   * distinct count values that are each greater than or equal to zero but less than max.
   * 
   * @param max
   * @param count
   * @param forced
   * @return
   */
  public static List<Integer> selectAndSample(int max, int count, int forced) {
    if (count >= max) {
      throw new IllegalArgumentException("Count " + count + " must be less than max: " + max);
    }
    List<Integer> selected = new ArrayList<>();
    selected.add(forced);
    List<Integer> availableIndices = new ArrayList<>(max - 1);
    for (int i = 0; i < max; ++i) {
      if (i != forced) {
        availableIndices.add(i);
      }
    }
    Collections.shuffle(availableIndices);
    for (int i = 0; i < count; ++i) {
      selected.add(availableIndices.get(i));
    }
    return selected;
  }

  public static List<Integer> sampleAndSkip(int max, int count, int skip) {
    if (count >= max) {
      throw new IllegalArgumentException("Count " + max + " must be less than max: " + count);
    }
    List<Integer> selected = new ArrayList<>();
    List<Integer> availableIndices = new ArrayList<>(max - 1);
    for (int i = 0; i < max; ++i) {
      if (i != skip) {
        availableIndices.add(i);
      }
    }
    Collections.shuffle(availableIndices);
    for (int i = 0; i < count; ++i) {
      selected.add(availableIndices.get(i));
    }
    return selected;
  }

  /**
   * Returns a new matrix whos 0th row is the firstRowIndex'th row of this matrix, and that has
   * randomRows other randomly selected rows.
   * 
   * @param firstRowIndex
   * @param randomRows
   * @return
   */
  public default ArrayMatrixValue<VarKey> selectAndSampleRowsWithElimination(int firstRowIndex,
      int randomRows) {
    return getRows(selectAndSample(getHeight(), randomRows, firstRowIndex));
  }

  public default ArrayMatrixValue<VarKey> selectAndSampleColumnsWithElimination(int firstIndex,
      int randomRows) {
    if (getWidth() < randomRows + 1) {
      throw new IllegalArgumentException("Width " + getWidth()
          + " must be equal at least to one plus randomRows: " + randomRows);
    }
    return getColumns(selectAndSample(getWidth(), randomRows, firstIndex));
  }

  public default ArrayMatrixValue<VarKey> getRows(List<Integer> rows) {
    Builder<VarKey> builder = new Builder<>(rows.size(), getWidth());
    for (int i = 0; i < rows.size(); ++i) {
      int inputRow = rows.get(i);
      for (int col = 0; col < getWidth(); ++col) {
        builder.set(i, col, get(inputRow, col));
      }
    }
    return builder.build();
  }

  public default ArrayMatrixValue<VarKey> getColumns(List<Integer> columns) {
    Builder<VarKey> builder = new Builder<>(getHeight(), columns.size());
    for (int i = 0; i < columns.size(); ++i) {
      int inputColumn = columns.get(i);
      for (int row = 0; row < getHeight(); ++row) {
        builder.set(row, i, get(row, inputColumn));
      }
    }
    return builder.build();
  }

  public default ArrayMatrixValue<VarKey> softmax() {
    if (getWidth() != 1) {
      throw new RuntimeException("Softmax supported only on matrices of width 1.");
    }
    Scalar<VarKey> max = get(0, 0);
    for (int i = 1; i < getHeight(); ++i) {
      if (get(i, 0).value() > max.value()) {
        max = get(i, 0);
      }
    }
    max = max.toConstant();

    List<Scalar<VarKey>> exp = new ArrayList<>();
    for (int i = 0; i < getHeight(); ++i) {
      exp.add(get(i, 0).minus(max).exp());
    }
    List<Scalar<VarKey>> denomComponents = new ArrayList<>();

    for (Scalar<VarKey> calc : exp) {
      denomComponents.add(calc);
    }
    Scalar<VarKey> sum = new StreamingSum<>(denomComponents).cache();
    Builder<VarKey> resultBuilder = new Builder<>(getHeight(), 1);
    for (int i = 0; i < exp.size(); ++i) {
      resultBuilder.set(i, 0, exp.get(i).divide(sum));
    }
    return resultBuilder.build();
  }

  public default Scalar<VarKey> maxIdx() {
    if (this.getWidth() != 1) {
      throw new RuntimeException("getMaxRowValue may only be used on 1-width matrices.");
    }
    double maxVal = get(0, 0).value();
    int maxIdx = 0;
    for (int i = 1; i < getHeight(); ++i) {
      double current = get(i, 0).value();
      if (get(i, 0).value() > maxVal) {
        maxVal = current;
        maxIdx = i;
      }
    }
    return Scalar.constant(maxIdx);
  }

  public default Matrix<VarKey> transform(Function<Scalar<VarKey>, Scalar<VarKey>> transform) {
    ArrayMatrixValue.Builder<VarKey> builder =
        new ArrayMatrixValue.Builder<>(getHeight(), getWidth());
    for (int i = 0; i < getHeight(); ++i) {
      for (int j = 0; j < getWidth(); ++j) {
        builder.set(i, j, transform.apply(get(i, j)));
      }
    }
    return builder.build();
  }

  public default Matrix<VarKey> toConstant() {
    ArrayMatrixValue.Builder<VarKey> builder =
        new ArrayMatrixValue.Builder<>(getHeight(), getWidth());
    for (int i = 0; i < getHeight(); ++i) {
      for (int j = 0; j < getWidth(); ++j) {
        Scalar<VarKey> c = Scalar.constant(this.get(i, j).value());
        builder.set(i, j, c);
      }
    }
    return builder.build();
  }

  public default Matrix<VarKey> getColumn(Scalar<VarKey> column) {
    return new MatrixRangeView<>(this, 0, (int) column.value(), getHeight(), 1);
  }
}
