package angland.optimizer.var;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BinaryOperator;

public class DerivativeMap<VarKey> {

  private List<KeyedDerivative<VarKey>>[] buckets;
  private int entryCount = 0;
  private double loadFactor = 1;

  private List<KeyedDerivative<VarKey>> getBucket(IndexedKey<VarKey> key) {
    return buckets[Math.abs(key.hashCode()) % buckets.length];
  }

  private List<KeyedDerivative<VarKey>> getOrCreateBucket(IndexedKey<VarKey> key) {
    return buckets[Math.abs(key.hashCode()) % buckets.length];
  }

  private KeyedDerivative<VarKey> getFromBucket(IndexedKey<VarKey> key,
      List<KeyedDerivative<VarKey>> bucket) {
    if (bucket != null) {
      for (int i = 0; i < bucket.size(); ++i) {
        KeyedDerivative<VarKey> kd = bucket.get(i);
        if (kd.getKey().equals(key)) {
          return kd;
        }
      }
    }
    return null;
  }

  public KeyedDerivative<VarKey> get(IndexedKey<VarKey> key) {
    List<KeyedDerivative<VarKey>> bucket = getBucket(key);
    return getFromBucket(key, bucket);
  }

  @SuppressWarnings("unchecked")
  private void expandIfNeeded() {
    if (entryCount * loadFactor <= buckets.length) {
      List<KeyedDerivative<VarKey>>[] old = buckets;
      buckets = new List[old.length * 2];
      for (List<KeyedDerivative<VarKey>> bucket : old) {
        if (bucket != null) {
          for (KeyedDerivative<VarKey> kd : bucket) {

          }
        }
      }
    }
  }

  public void merge(IndexedKey<VarKey> key, double value, BinaryOperator<Double> operation) {
    List<KeyedDerivative<VarKey>> bucket = getBucket(key);
    if (bucket == null) {
      bucket = new ArrayList<>(1);
      buckets[Math.abs(key.hashCode()) % buckets.length] = bucket;
    }
    KeyedDerivative<VarKey> entry = getFromBucket(key, bucket);
    if (entry != null) {
      entry.setValue(operation.apply(entry.getValue(), value));
      return;
    }
    expandIfNeeded();
    bucket = getBucket(key);

  }



}
