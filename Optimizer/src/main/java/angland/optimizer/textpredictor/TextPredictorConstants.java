package angland.optimizer.textpredictor;

import angland.optimizer.nn.LstmCellTemplate;
import angland.optimizer.nn.RnnCellTemplate;

class TextPredictorConstants {
  static int vocabSize = 10000;
  static int lstmSize = 160;
  static double gradientClipThreshold = .01;

  public static RnnCellTemplate getTemplate(boolean constant) {
    return new LstmCellTemplate("cell", lstmSize, gradientClipThreshold, constant);
    /*
     * return FeatureGroupRnnCellTemplate.simple("cell", lstmSize, 4, gradientClipThreshold,
     * constant, true);
     */
  }


}
