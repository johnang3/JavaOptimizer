package angland.optimizer.var;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.BinaryOperator;
import java.util.function.Function;

import angland.optimizer.var.ArrayMatrixValue.Builder;
import angland.optimizer.var.scalar.IScalarValue;
import angland.optimizer.var.scalar.MappedDerivativeScalar;
import angland.optimizer.var.scalar.ScalarConstant;
import angland.optimizer.var.scalar.StreamingSum;

public interface IMatrixValue<VarKey> {

  public int getHeight();

  public int getWidth();

  public IScalarValue<VarKey> get(int row, int column);

  public static <VarKey> IMatrixValue<VarKey> var(VarKey key, int height, int width,
      Map<IndexedKey<VarKey>, Double> context) {
    ArrayMatrixValue.Builder<VarKey> builder = new ArrayMatrixValue.Builder<VarKey>(height, width);
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        IndexedKey<VarKey> indexedKey = IndexedKey.matrixKey(key, i, j);
        builder.set(i, j, IScalarValue.varIndexed(indexedKey, context));
      }
    }
    return builder.build();
  }

  public static <VarKey> IMatrixValue<VarKey> repeat(IScalarValue<VarKey> val, int height, int width) {
    ArrayMatrixValue.Builder<VarKey> builder = new ArrayMatrixValue.Builder<VarKey>(height, width);
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        builder.set(i, j, val);
      }
    }
    return builder.build();
  }

  public default MappedDerivativeScalar<VarKey> elementSum() {
    MappedDerivativeScalar.Builder<VarKey> sum = new MappedDerivativeScalar.Builder<>(1);
    for (int i = 0; i < getHeight(); ++i) {
      for (int j = 0; j < getWidth(); ++j) {
        sum.increment(get(i, j));
      }
    }
    return sum.build();
  }

  public default IMatrixValue<VarKey> pointwise(IMatrixValue<VarKey> other,
      BinaryOperator<IScalarValue<VarKey>> op) {
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

  public default IMatrixValue<VarKey> plus(IMatrixValue<VarKey> other) {
    return pointwise(other, IScalarValue::plus);
  }


  public default IMatrixValue<VarKey> pointwiseMultiply(IMatrixValue<VarKey> other) {
    return pointwise(other, IScalarValue::times);
  }

  public default IMatrixValue<VarKey> columnProximity(IMatrixValue<VarKey> other) {
    List<Integer> list = new ArrayList<>();
    for (int i = 0; i < getWidth(); ++i) {
      list.add(i);
    }
    return columnProximity(other, list);
  }

  public default IMatrixValue<VarKey> columnProximity(IMatrixValue<VarKey> other,
      List<Integer> selectedIndices) {
    if (this.getHeight() != other.getHeight()) {
      throw new IllegalArgumentException("Cannot compare columns in matrices of differing heights");
    }
    if (other.getWidth() != 1) {
      throw new IllegalArgumentException(
          "Can only compare column proximity to a matrix of width 1.");
    }
    Builder<VarKey> newMatrix = new Builder<>(1, getWidth());
    for (int i = 0; i < getWidth(); ++i) {
      newMatrix.set(0, i, IScalarValue.constant(Double.MAX_VALUE));
    }
    for (int i : selectedIndices) {
      List<IScalarValue<VarKey>> components = new ArrayList<>(getWidth());
      for (int j = 0; j < getHeight(); ++j) {
        components.add(this.get(j, i).minus(other.get(j, 0)).power(2));
      }
      newMatrix.set(0, i, new StreamingSum<>(components).power(.5).cache());
      // KeyedDerivative.printRelativeDist(newMatrix.get(0, i).getGradient());
    }
    return newMatrix.build();
  }

  public default IMatrixValue<VarKey> sampledColumnProximity(IMatrixValue<VarKey> other,
      int requiredIndex, int samples) {
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
    IMatrixValue<VarKey> m = columnProximity(other, selected);
    return m;
  }

  public default IMatrixValue<VarKey> times(IMatrixValue<VarKey> other) {
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
          IScalarValue<VarKey> left = this.get(i, k);
          IScalarValue<VarKey> right = other.get(k, j);
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
        builder.set(i, j, sumBuilder.build());
      }
    }
    return builder.build();
  }


  public default IMatrixValue<VarKey> transpose() {
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
    IScalarValue<VarKey> zero = IScalarValue.constant(0);
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

  public default ArrayMatrixValue<VarKey> softmax() {
    if (getWidth() != 1) {
      throw new RuntimeException("Softmax supported only on matrices of width 1.");
    }
    List<IScalarValue<VarKey>> exp = new ArrayList<>();
    for (int i = 0; i < getHeight(); ++i) {
      exp.add(get(i, 0).exp());
    }
    List<IScalarValue<VarKey>> denomComponents = new ArrayList<>();

    for (IScalarValue<VarKey> calc : exp) {
      denomComponents.add(calc);
    }
    IScalarValue<VarKey> sum = new StreamingSum<>(denomComponents).cache();
    Builder<VarKey> resultBuilder = new Builder<>(getHeight(), 1);
    for (int i = 0; i < exp.size(); ++i) {
      resultBuilder.set(i, 0, exp.get(i).divide(sum));
    }
    return resultBuilder.build();
  }

  public default IScalarValue<VarKey> maxIdx() {
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
    return new ScalarConstant<>(maxIdx);
  }

  public default IMatrixValue<VarKey> transform(
      Function<IScalarValue<VarKey>, IScalarValue<VarKey>> transform) {
    ArrayMatrixValue.Builder<VarKey> builder =
        new ArrayMatrixValue.Builder<>(getHeight(), getWidth());
    for (int i = 0; i < getHeight(); ++i) {
      for (int j = 0; j < getWidth(); ++j) {
        builder.set(i, j, transform.apply(get(i, j)));
      }
    }
    return builder.build();
  }

  public default IMatrixValue<VarKey> toConstant() {
    ArrayMatrixValue.Builder<VarKey> builder =
        new ArrayMatrixValue.Builder<>(getHeight(), getWidth());
    for (int i = 0; i < getHeight(); ++i) {
      for (int j = 0; j < getWidth(); ++j) {
        IScalarValue<VarKey> c = IScalarValue.constant(this.get(i, j).value());
        builder.set(i, j, c);
      }
    }
    return builder.build();
  }

  public default IMatrixValue<VarKey> getColumn(IScalarValue<VarKey> column) {
    return new MatrixRangeView<>(this, 0, (int) column.value(), getHeight(), 1);
  }
}
