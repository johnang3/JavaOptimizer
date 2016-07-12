package angland.optimizer.utils;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Stream;

public class ObjectToDoubleMap<Key> {

  private List<Entry<Key>>[] arr;
  private final double loadFactor;
  private int size;

  private ObjectToDoubleMap(List<Entry<Key>>[] arr, double loadFactor, int size) {
    this.arr = arr;
    this.loadFactor = loadFactor;
    this.size = size;
  }

  @SuppressWarnings("unchecked")
  public ObjectToDoubleMap(int initialSize, double loadFactor) {
    this(new List[initialSize], loadFactor, 0);
  }

  @SuppressWarnings("unchecked")
  public ObjectToDoubleMap(int initialSize) {
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
    return arr[Math.abs(key.hashCode()) % arr.length];
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


  public void put(Key key, double value) {
    int absHashCode = Math.abs(key.hashCode());
    int bucketIdx = absHashCode % arr.length;
    List<Entry<Key>> bucket = arr[bucketIdx];
    Entry<Key> entry = getEntryFromBucket(key, bucket);
    if (entry == null) {
      entry = new Entry<Key>(key, value, absHashCode);
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
            List<Entry<Key>> targetBucket = a2[oldEntry.hashCode() % a2.length];
            if (targetBucket == null) {
              targetBucket = new ArrayList<>(1);
              a2[oldEntry.hashCode() % a2.length] = targetBucket;
            }
            targetBucket.add(oldEntry);
          }
        }
      }
      this.arr = a2;
      existingBucketIdx = entry.hashCode() % arr.length;
      existingBucket = arr[existingBucketIdx];
      if (existingBucket == null) {
        existingBucket = new ArrayList<>(1);
        arr[existingBucketIdx] = existingBucket;
      }
      existingBucket.add(entry);
    }
  }

  public void adjust(Key key, double shift) {
    int absHashCode = Math.abs(key.hashCode());
    int bucketIdx = absHashCode % arr.length;
    List<Entry<Key>> bucket = arr[bucketIdx];
    Entry<Key> entry = getEntryFromBucket(key, bucket);
    if (entry != null) {
      entry.value += shift;
    } else {
      entry = new Entry<>(key, shift, absHashCode);
      insertNewValue(bucketIdx, bucket, entry);
    }
  }

  public ObjectToDoubleMap<Key> cloneWithMultiplier(double multiplier) {
    @SuppressWarnings("unchecked")
    List<Entry<Key>>[] a2 = new List[arr.length];
    for (int i = 0; i < arr.length; ++i) {
      List<Entry<Key>> bucket = arr[i];
      if (bucket != null) {
        a2[i] = new ArrayList<>(bucket.size());
        for (Entry<Key> entry : bucket) {
          a2[i].add(new Entry<>(entry.key, entry.value * multiplier, entry.hashCode()));
        }
      }
    }
    return new ObjectToDoubleMap<>(a2, loadFactor, size);
  }

  public void actOnEntries(Consumer<Entry<Key>> consumer) {
    for (List<Entry<Key>> bucket : arr) {
      if (bucket != null) {
        for (Entry<Key> entry : bucket) {
          consumer.accept(entry);
        }
      }
    }
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
    private double value;
    private final int hashCode;

    public Entry(Key key, double value, int hashCode) {
      super();
      this.key = key;
      this.value = value;
      this.hashCode = hashCode;
    }

    public Key getKey() {
      return key;
    }

    public double getValue() {
      return value;
    }

    public void setValue(double value) {
      this.value = value;
    }

    @Override
    public int hashCode() {
      return hashCode;
    }
  }

}
