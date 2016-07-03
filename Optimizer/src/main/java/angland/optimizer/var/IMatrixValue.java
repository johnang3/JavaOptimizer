package angland.optimizer.var;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import angland.optimizer.var.ArrayMatrixValue.Builder;

public interface IMatrixValue<VarKey> {

  public int getHeight();

  public int getWidth();

  public ScalarValue<VarKey> get(int row, int column);

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
      for (int j = 0; j < getWidth(); ++j) {
        newMatrix.set(i, j, this.get(i, j).plus(other.get(i, j)));
      }
    }
    return newMatrix.build();
  }


  public default IMatrixValue<VarKey> multiplyPointwise(IMatrixValue<VarKey> other) {
    if (this.getHeight() != other.getHeight()) {
      throw new RuntimeException("Cannot add matrices of differing heights.");
    }
    if (this.getWidth() != other.getWidth()) {
      throw new RuntimeException("Cannot add matrices of differing widths.");
    }
    Builder<VarKey> newMatrix = new Builder<>(getHeight(), getWidth());
    for (int i = 0; i < getHeight(); ++i) {
      for (int j = 0; j < getWidth(); ++j) {
        newMatrix.set(i, j, this.get(i, j).times(other.get(i, j)));
      }
    }
    return newMatrix.build();
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
          ScalarValue<VarKey> left = this.get(i, k);
          ScalarValue<VarKey> right = other.get(k, j);
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

  public default ArrayMatrixValue<VarKey> softmax() {
    if (getWidth() != 1) {
      throw new RuntimeException("Softmax supported only on matrices of width 1.");
    }
    List<ScalarValue<VarKey>> exp = new ArrayList<>();
    for (int i = 0; i < getHeight(); ++i) {
      exp.add(get(i, 0).exp());
    }
    ScalarValue.Builder<VarKey> sumBuilder = new ScalarValue.Builder<>(exp.size());
    for (ScalarValue<VarKey> calc : exp) {
      sumBuilder.increment(calc);
    }
    ScalarValue<VarKey> sum = sumBuilder.build();
    Builder<VarKey> resultBuilder = new Builder<>(getHeight(), 1);
    for (int i = 0; i < exp.size(); ++i) {
      resultBuilder.set(i, 0, exp.get(i).divide(sum));
    }
    return resultBuilder.build();
  }

  public default ScalarValue<VarKey> maxIdx() {
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
    return new ScalarValue<>(maxIdx, new HashMap<>());
  }

  public default IMatrixValue<VarKey> transform(
      Function<ScalarValue<VarKey>, ScalarValue<VarKey>> transform) {
    ArrayMatrixValue.Builder<VarKey> builder =
        new ArrayMatrixValue.Builder<>(getHeight(), getWidth());
    for (int i = 0; i < getHeight(); ++i) {
      for (int j = 0; j < getWidth(); ++j) {
        builder.set(i, j, transform.apply(get(i, j)));
      }
    }
    return builder.build();
  }

  public default MatrixExpression<VarKey> toConstant() {
    return (ctx, cache) -> {
      ArrayMatrixValue.Builder<VarKey> builder =
          new ArrayMatrixValue.Builder<>(getHeight(), getWidth());
      for (int i = 0; i < getHeight(); ++i) {
        for (int j = 0; j < getWidth(); ++j) {
          ScalarExpression<VarKey> c = ScalarExpression.constant(this.get(i, j).value());
          builder.set(i, j, c.evaluate(ctx));
        }
      }
      return builder.build();
    };
  }
}