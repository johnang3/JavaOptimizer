package angland.optimizer.utils;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

public class ObjectToFloatMap<Key> {

  private List<Entry<Key>>[] arr;
  private final double loadFactor;
  private int size;

  private ObjectToFloatMap(List<Entry<Key>>[] arr, double loadFactor, int size) {
    this.arr = arr;
    this.loadFactor = loadFactor;
    this.size = size;
  }

  @SuppressWarnings("unchecked")
  public ObjectToFloatMap(int initialSize, double loadFactor) {
    this(new List[initialSize], loadFactor, 0);
  }

  @SuppressWarnings("unchecked")
  public ObjectToFloatMap(int initialSize) {
    this(new List[initialSize], 1.0, 0);
  }

  public int size() {
    return (int) size;
  }

  private Entry<Key> getEntry(Key key) {
    List<Entry<Key>> bucket = getBucket(key);
    return getEntryFromBucket(key, bucket);
  }

  private List<Entry<Key>> getBucket(Key key) {
    return arr[key.hashCode() % arr.length];
  }

  private Entry<Key> getEntryFromBucket(Key key, List<Entry<Key>> bucket) {
    if (bucket != null) {
      for (Entry<Key> entry : bucket) {
        if (entry.getKey().equals(key)) {
          return entry;
        }
      }
    }
    return null;
  }

  /**
   * Returns the associated value for the specified key, or zero if no entry is present.
   * 
   * @param key
   * @return
   */
  public double get(Key key) {
    Entry<Key> entry = getEntry(key);
    return entry == null ? 0 : entry.getValue();
  }

  public void put(Key key, float value) {
    int bucketIdx = key.hashCode() % arr.length;
    List<Entry<Key>> bucket = arr[bucketIdx];
    Entry<Key> entry = getEntryFromBucket(key, bucket);
    if (entry == null) {
      entry = new Entry<Key>(key, value);
      insertNewValue(bucketIdx, bucket, entry);
    }
    entry.value = value;
  }

  private void insertNewValue(int existingBucketIdx, List<Entry<Key>> existingBucket,
      Entry<Key> entry) {
    ++size;
    if (size <= loadFactor * arr.length) {
      if (existingBucket == null) {
        existingBucket = new ArrayList<>(1);
        arr[existingBucketIdx] = existingBucket;
      }
      existingBucket.add(entry);
    } else {
      @SuppressWarnings("unchecked")
      List<Entry<Key>>[] a2 = new List[arr.length * 2];
      for (List<Entry<Key>> oldBucket : arr) {
        if (oldBucket != null) {
          for (Entry<Key> oldEntry : oldBucket) {
            List<Entry<Key>> targetBucket = a2[oldEntry.getKey().hashCode() % a2.length];
            if (targetBucket == null) {
              targetBucket = new ArrayList<>(1);
              a2[oldEntry.getKey().hashCode() % a2.length] = targetBucket;
            }
            targetBucket.add(oldEntry);
          }
        }
      }
      this.arr = a2;
      existingBucketIdx = entry.getKey().hashCode() % arr.length;
      existingBucket = arr[existingBucketIdx];
      if (existingBucket == null) {
        existingBucket = new ArrayList<>(1);
        arr[existingBucketIdx] = existingBucket;
      }
      existingBucket.add(entry);
    }
  }

  public void adjust(Key key, float shift) {
    int bucketIdx = key.hashCode() % arr.length;
    List<Entry<Key>> bucket = arr[bucketIdx];
    Entry<Key> entry = getEntryFromBucket(key, bucket);
    if (entry != null) {
      entry.value += shift;
    } else {
      entry = new Entry<>(key, shift);
      insertNewValue(bucketIdx, bucket, entry);
    }
  }

  public ObjectToFloatMap<Key> cloneWithMultiplier(float multiplier) {
    @SuppressWarnings("unchecked")
    List<Entry<Key>>[] a2 = new List[arr.length];
    for (int i = 0; i < arr.length; ++i) {
      List<Entry<Key>> bucket = arr[i];
      if (bucket != null) {
        a2[i] = new ArrayList<>(bucket.size());
        for (Entry<Key> entry : bucket) {
          a2[i].add(new Entry<>(entry.key, entry.value * multiplier));
        }
      }
    }
    return new ObjectToFloatMap<>(a2, loadFactor, size);
  }

  public Stream<Entry<Key>> entries() {
    Stream<Entry<Key>> s = Stream.empty();
    for (List<Entry<Key>> bucket : arr) {
      if (bucket != null) {
        s = Stream.concat(s, bucket.stream());
      }
    }
    return s;
  }

  public static class Entry<Key> {
    private final Key key;
    private float value;

    public Entry(Key key, float value) {
      super();
      this.key = key;
      this.value = value;
    }

    public Key getKey() {
      return key;
    }

    public double getValue() {
      return value;
    }

    public void setValue(float value) {
      this.value = value;
    }
  }

}
