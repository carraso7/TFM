FUNCTION
Calculate the return period of any hydrographs. 
Method: The return Period of Flood Flows, E. J. Gumbel (1941).The Annals of Mathematical Statistics, Vol. 12, No. 2,pp. 163-190

INPUTS
streamflow: Import a .txt where rows have to be in datetime type and columns have to be the daily flow of each gauging station.
            (Default: 'aforo_subcuencas_ebro.txt')
stations: Stations library (aforo_subcuencas_ebro) contains all stations included in SAIH web for north face Ebro. 
          You can add all number stations you need in a string list (e.g. ['001','002','003', '018'])
          If you have imported another .txt, you have to include elements of new columns.
          Excel document 'Tabla.txt' has the information about the name station asociated to the number station.          
returnperiod: You can add a integer list with the return period you want to calculate (e.g. [0.5, 1, 2, 5, 10] [years]) 

OUTPUTS
popt: first element is the 'a' element in the logarithm function
      second element is the 'b' element in the logarithm function
      Being logarithm function: y=a*ln(x)+b  where x is the return period.
	  The scrip prints function with elements fro each station.
y_p: Dictionary with each streamflow for each return period and each station.
	 The script prints the return periods for each station.
figure 1: plot log curve. This curve is the trend of the maximum annual data observed. (x axis: return period, y axis: streamflow) 
figure 2: plot daily hydrograph with desired return periods