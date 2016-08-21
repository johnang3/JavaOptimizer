package angland.optimizer.var;

import angland.optimizer.Range;

public class IndexedKeyRange<VarKey> {

  private final IndexedKey<VarKey> indexedKey;
  private final Range range;

  public IndexedKeyRange(IndexedKey<VarKey> indexedKey, Range range) {
    super();
    this.indexedKey = indexedKey;
    this.range = range;
  }

  public IndexedKey<VarKey> getIndexedKey() {
    return indexedKey;
  }

  public Range getRange() {
    return range;
  }



}
