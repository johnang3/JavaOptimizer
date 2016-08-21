package angland.optimizer.var;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;



public class DerivativeMap<VarKey> {


  private KeyedDerivativeMapNode<VarKey>[] buckets;
  private int entryCount = 0;
  private float loadFactor = 1;

  @SuppressWarnings("unchecked")
  public DerivativeMap(int initialArraySize) {
    this.buckets = new KeyedDerivativeMapNode[initialArraySize];
  }

  public double get(IndexedKey<VarKey> key) {
    KeyedDerivativeMapNode<VarKey> node = buckets[Math.abs(key.hashCode()) % buckets.length];
    while (node != null) {
      if (key.equals(node.getKey())) {
        return node.value;
      }
      node = node.next;
    }
    return 0;
  }

  @SuppressWarnings("unchecked")
  private void reorgIfNeeded() {
    if (loadFactor * entryCount > buckets.length) {
      KeyedDerivativeMapNode<VarKey>[] oldBuckets = buckets;
      buckets = new KeyedDerivativeMapNode[oldBuckets.length * 2];
      for (int i = 0; i < oldBuckets.length; ++i) {
        KeyedDerivativeMapNode<VarKey> node = oldBuckets[i];
        while (node != null) {
          KeyedDerivativeMapNode<VarKey> copy =
              new KeyedDerivativeMapNode<>(node.getKey(), node.getValue());
          int idx = Math.abs(copy.getKey().hashCode()) % buckets.length;
          copy.next = buckets[idx];
          buckets[idx] = copy;
          node = node.next;
        }
      }
    }
  }

  public void merge(IndexedKey<VarKey> key, double value) {
    KeyedDerivativeMapNode<VarKey> node = buckets[Math.abs(key.hashCode()) % buckets.length];
    while (node != null) {
      if (node.getKey().equals(key)) {
        node.setValue(node.getValue() + value);
        return;
      }
      node = node.next;
    }
    KeyedDerivativeMapNode<VarKey> newNode = new KeyedDerivativeMapNode<>(key, value);
    newNode.next = buckets[Math.abs(key.hashCode()) % buckets.length];
    buckets[Math.abs(key.hashCode()) % buckets.length] = newNode;
    ++entryCount;
    reorgIfNeeded();
  }

  public void actOnEntries(Consumer<KeyedDerivative<VarKey>> consumer) {
    for (int i = 0; i < buckets.length; ++i) {
      KeyedDerivativeMapNode<VarKey> node = buckets[i];
      while (node != null) {
        consumer.accept(node);
        node = node.next;
      }
    }
  }

  /**
   * 
   * If n is greater than or equal to the number of entries, returns this map.
   * 
   * Otherwise, returns a new derivativemap containing only the n entries from this map with the
   * highest absolute value.
   * 
   * @return
   */
  public DerivativeMap<VarKey> getHighestAbs(int n) {
    if (n >= entryCount) return this;
    List<KeyedDerivative<VarKey>> entries = new ArrayList<>();
    actOnEntries(entries::add);
    Collections.sort(entries, (e1, e2) -> Double.compare(Math.abs(e2.value), Math.abs(e1.value)));
    DerivativeMap<VarKey> outputMap = new DerivativeMap<>(n);
    for (int i = 0; i < n; ++i) {
      outputMap.merge(entries.get(i).getKey(), entries.get(i).value);
    }
    return outputMap;
  }

}
