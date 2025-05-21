* SAS Program: import_explore.sas ;
* Purpose: Import Airbnb data and perform initial exploration & formatting.;

* Set a libname for where to store the SAS dataset (optional, can use WORK library too) ;
* For this example, we'll use the WORK library which is temporary. ;
* LIBNAME project "." ; * This would create a permanent library in the current directory. For now, WORK is fine.;

* 1. Import the CSV file using DATA step with explicit variables ;
FILENAME ABDATA "/home/u64208646/Project/featured_AB_NYC_2019.csv";

DATA WORK.nyc_airbnb;
    /* Use efficient INFILE options */
    INFILE ABDATA DLM=',' MISSOVER DSD FIRSTOBS=2 TRUNCOVER;

    /* INPUT statement with explicit variables */
    INPUT
        id $
        name $
        host_id $
        host_name $
        neighbourhood_group $
        neighbourhood $
        latitude
        longitude
        room_type $
        price
        minimum_nights
        number_of_reviews
        last_review $
        reviews_per_month
        calculated_host_listings_count
        availability_365
        has_reviews
        days_since_last_review;
RUN;

FILENAME ABDATA CLEAR;

TITLE "Contents of Imported Data (Check Column Types)";
PROC CONTENTS DATA=WORK.nyc_airbnb;
RUN;

TITLE "First 20 Rows of Imported Data (Check Data Alignment)";
PROC PRINT DATA=WORK.nyc_airbnb (OBS=20);
    VAR id name host_id host_name neighbourhood_group neighbourhood latitude longitude room_type price minimum_nights number_of_reviews last_review reviews_per_month calculated_host_listings_count availability_365 has_reviews days_since_last_review;
RUN;

TITLE "Descriptive Statistics for Numerical Variables";
PROC MEANS DATA=WORK.nyc_airbnb N MEAN STD MIN MAX Q1 MEDIAN Q3;
    VAR price minimum_nights number_of_reviews reviews_per_month calculated_host_listings_count availability_365 days_since_last_review;
RUN;

TITLE "Frequency Counts for Key Categorical Variables (Original Values)";
PROC FREQ DATA=WORK.nyc_airbnb ORDER=FREQ;
    TABLES neighbourhood_group room_type has_reviews / MISSING;
RUN;

* 3. Create User-Defined Formats ;
TITLE "Creating and Applying User-Defined Formats";
PROC FORMAT;
    VALUE $neigh_group_fmt
        'Bronx'         = 'Bronx'
        'Brooklyn'      = 'Brooklyn'
        'Manhattan'     = 'Manhattan'
        'Queens'        = 'Queens'
        'Staten Island' = 'Staten Island'
        OTHER           = 'Other/Unknown';

    VALUE $room_type_fmt
        'Entire home/apt' = 'Entire Home/Apt'
        'Private room'    = 'Private Room'
        'Shared room'     = 'Shared Room'
        OTHER             = 'Other/Unknown';

    VALUE review_fmt /* Numeric format for has_reviews */
        0 = 'No Reviews'
        1 = 'Has Reviews'
        . = 'Missing'
        OTHER = 'Error';
RUN;

* Apply formats in a new data step for clarity and further processing ;
DATA WORK.airbnb_processed;
    SET WORK.nyc_airbnb;

    FORMAT neighbourhood_group $neigh_group_fmt.
           room_type $room_type_fmt.
           has_reviews review_fmt.;

    * 4. Data Step Enhancements: Conditional Processing & New Variables ;
    * Create price_category ;
    IF price <= 75 THEN price_category = 'Budget';
    ELSE IF 75 < price <= 200 THEN price_category = 'Mid-Range';
    ELSE IF price > 200 THEN price_category = 'Premium';
    ELSE price_category = 'Unknown'; /* Should not happen if price is not missing */
    LABEL price_category = "Price Category";

    * Extract year_last_review (ensure last_review is handled correctly) ;
    * Using a more straightforward approach to extract year from last_review;
    year_last_review = .;
    IF NOT MISSING(last_review) THEN
        year_last_review = YEAR(INPUT(SUBSTR(last_review,1,10), B8601DA10.));
    LABEL year_last_review = "Year of Last Review";
    FORMAT year_last_review 4.;

RUN;

TITLE "Frequency of New Price Categories";
PROC FREQ DATA=WORK.airbnb_processed ORDER=FREQ;
    TABLES price_category;
RUN;

TITLE "Frequency of Year of Last Review";
PROC FREQ DATA=WORK.airbnb_processed ORDER=FREQ;
    TABLES year_last_review / MISSING; /* See distribution of review years */
RUN;


* Create a subset: Manhattan listings with high availability ;
DATA WORK.manhattan_high_avail;
    SET WORK.airbnb_processed;
    WHERE neighbourhood_group = 'Manhattan' AND availability_365 > 180;
RUN;

TITLE "Manhattan Listings with Availability > 180 days";
PROC PRINT DATA=WORK.manhattan_high_avail (OBS=10);
    VAR name neighbourhood_group room_type price availability_365;
RUN;
PROC MEANS DATA=WORK.manhattan_high_avail NOPRINT;
    OUTPUT OUT=WORK.manhattan_stats (DROP=_TYPE_ _FREQ_) N=count_manhattan_high_avail;
RUN;
PROC PRINT DATA=WORK.manhattan_stats;
    TITLE "Count of Manhattan Listings with Availability > 180 days";
RUN;


* 5. PROC SQL Query ;
TITLE "Average Price and Count by Neighbourhood Group and Room Type";
PROC SQL;
    SELECT neighbourhood_group,
           room_type,
           COUNT(*) AS listing_count FORMAT=COMMA10.,
           AVG(price) AS average_price FORMAT=DOLLAR8.2
    FROM WORK.airbnb_processed
    GROUP BY neighbourhood_group, room_type
    ORDER BY neighbourhood_group, listing_count DESC;
QUIT;

* 6. PROC REPORT ;
TITLE "Listings Report for Brooklyn (Sample)";
PROC REPORT DATA=WORK.airbnb_processed(OBS=100) NOWD HEADSKIP;
    COLUMN name neighbourhood_group room_type price number_of_reviews availability_365;
    DEFINE name / DISPLAY "Listing Name" WIDTH=40;
    DEFINE neighbourhood_group / DISPLAY "Borough";
    DEFINE room_type / ORDER "Room Type";
    DEFINE price / ANALYSIS MEAN "Avg Price" FORMAT=DOLLAR8.2; /* Will show overall mean if no group */
    DEFINE number_of_reviews / ANALYSIS SUM "Total Reviews";
    DEFINE availability_365 / DISPLAY "Days Available";

    WHERE neighbourhood_group = 'Brooklyn'; /* Filter for Brooklyn */

    /* Example of breaking by room_type within Brooklyn and showing summary stats */
    BREAK AFTER room_type / SUMMARIZE STYLE=Header{font_weight=bold};
    RBREAK AFTER / SUMMARIZE STYLE=Header{font_weight=bold textalign=right}; /* Grand total summary */

    COMPUTE AFTER room_type;
        name = "Subtotal for " || room_type;
    ENDCOMP;
    COMPUTE AFTER;
        name = "Grand Total for Brooklyn";
    ENDCOMP;
RUN;
TITLE;


* 7. PROC SGPLOT - Graphics ;
TITLE "Average Price by Neighbourhood Group (SGPLOT)";
PROC SGPLOT DATA=WORK.airbnb_processed;
    VBAR neighbourhood_group / RESPONSE=price STAT=MEAN
        DATALABEL DATALABELATTRS=(color=gray) GROUPDISPLAY=CLUSTER;
    YAXIS LABEL="Average Price ($)";
    XAXIS LABEL="Neighbourhood Group";
RUN;
TITLE;

TITLE "Price vs. Number of Reviews (SGPLOT)";
PROC SGPLOT DATA=WORK.airbnb_processed;
    SCATTER X=number_of_reviews Y=price / GROUP=neighbourhood_group TRANSPARENCY=0.5;
    YAXIS LABEL="Price ($)";
    XAXIS LABEL="Number of Reviews";
    KEYLEGEND / LOCATION=OUTSIDE POSITION=BOTTOM TITLE=""; /* Adjust legend */
    /* Add a regression line for overall trend */
    REG X=number_of_reviews Y=price / NOMARKERS LINEATTRS=(color=red thickness=2);
RUN;
TITLE;


* 8. PROC REG - Simple Linear Regression ;
TITLE "Simple Linear Regression: Price vs. Number of Reviews";
PROC REG DATA=WORK.airbnb_processed PLOTS(MAXPOINTS=NONE);
    MODEL price = number_of_reviews;
    /* Add more predictors here if desired, e.g., number_of_reviews minimum_nights */
    /* PLOT R.*P.; */ /* Example for residual plots */
RUN;
QUIT;

* 9. PROC LOGISTIC - Predicting 'has_reviews' (SAS ML Example) ;
TITLE "Logistic Regression: Predicting if a Listing Has Reviews";
PROC LOGISTIC DATA=WORK.airbnb_processed DESCENDING; /* DESCENDING to model P(has_reviews=1) */
    CLASS neighbourhood_group room_type / PARAM=REF REF=FIRST;
    MODEL has_reviews = price minimum_nights calculated_host_listings_count availability_365
                        neighbourhood_group room_type / SELECTION=FORWARD LINK=LOGIT TECHNIQUE=FISHER RSQUARE;
    /* SELECTION=FORWARD for variable selection example */
    /* TECHNIQUE=FISHER can be more stable for some datasets */
    /* RSQUARE for pseudo R-squared measures */
RUN;
QUIT;

TITLE; /* Clear title */

* End of program. ;