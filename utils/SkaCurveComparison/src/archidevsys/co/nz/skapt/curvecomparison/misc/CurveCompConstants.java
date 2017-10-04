package archidevsys.co.nz.skapt.curvecomparison.misc;

public interface CurveCompConstants
{
    int KB = 1024;

    int MB = 1024 * KB;;

    int BUFFER_SIZE = 1 * MB;

    int DATA_ROW_COUNT_ESTIMATE = 1 * MB;

    int FREQUENCY_CHANNEL_COUNT_ESTIMATE = 4 * KB;

    int ROW_PART_COUNT = 7;

    int STOKES_VECTOR_LENGTH = 4;

    /*
     * See: Summary of regular-expression constructs in the Pattern class
     * 
     * Accepts one or more spaces
     * 
     */
    String REGEX_SPACES = "(\\h)+";

    String YYYYMMDD_HHMMSS_FORMAT = "%1$tY%1$tm%1$td_%1$tH%1$tM%1$tS";

    String EMPTY_STRING = "";

    String DOUBLE_QUOTES = "\"";

    String DEFAULT_LOG_PREFIX = "curveComparison-";
}
