package angland.optimizer.var;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

public class MatrixOld<RowKey, ColumnKey> {

  private Map<RowKey, Map<ColumnKey, Double>> rowView = new HashMap<>(2);
  private Map<ColumnKey, Map<RowKey, Double>> columnView = new HashMap<>(2);

  public Set<RowKey> getRowKeys() {
    return rowView.keySet();
  }

  public Set<ColumnKey> getColumnKeys() {
    return columnView.keySet();
  }

  public double get(RowKey rowKey, ColumnKey columnKey) {
    return rowView.computeIfAbsent(rowKey, rk -> new HashMap<>(2)).getOrDefault(columnKey, 0d);
  }

  public void put(RowKey rowKey, ColumnKey columnKey, double val) {
    rowView.computeIfAbsent(rowKey, rk -> new HashMap<>(2)).put(columnKey, val);
    columnView.computeIfAbsent(columnKey, ck -> new HashMap<>(2)).put(rowKey, val);
  }

  public void increment(RowKey rowKey, ColumnKey columnKey, double val) {
    rowView.computeIfAbsent(rowKey, rk -> new HashMap<>(2)).merge(columnKey, val, Double::sum);
    columnView.computeIfAbsent(columnKey, ck -> new HashMap<>(2)).merge(rowKey, val, Double::sum);
  }

  public Stream<Entry> getAllEntries() {
    return rowView
        .entrySet()
        .stream()
        .flatMap(
            rc -> rc.getValue().entrySet().stream()
                .map(e -> new Entry(rc.getKey(), e.getKey(), e.getValue())));
  }

  public static <X, Y, Z> MatrixOld<X, Z> multiply(MatrixOld<X, Y> left, MatrixOld<Y, Z> right) {
    if (!left.getColumnKeys().equals(right.getRowKeys())) {
      throw new RuntimeException(
          "Left matrix column keySet must be equal to right matrix row keySet.");
    }
    MatrixOld<X, Z> product = new MatrixOld<>();
    for (X rowKey : left.getRowKeys()) {
      for (Z colKey : right.getColumnKeys()) {
        for (Y y : left.getColumnKeys()) {
          product.increment(rowKey, colKey, left.get(rowKey, y) * right.get(y, colKey));
        }
      }
    }
    return product;
  }

  public class Entry {
    private final RowKey rowKey;
    private final ColumnKey columnKey;
    private final double value;

    public Entry(RowKey rowKey, ColumnKey columnKey, double value) {
      super();
      this.rowKey = rowKey;
      this.columnKey = columnKey;
      this.value = value;
    }

    public RowKey getRowKey() {
      return rowKey;
    }

    public ColumnKey getColumnKey() {
      return columnKey;
    }

    public double getValue() {
      return value;
    }


  }



}
