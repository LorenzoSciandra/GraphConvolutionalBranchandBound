/*********************************************
 * OPL 22.1.0.0 Model
 * Author: lorenzosciandra
 * Creation Date: 18 May 2022 at 19:59:31
 *********************************************/

// Data
int n = ...;
range cities = 1 .. n;
int Cost[cities][cities] = ...;


// Decision variables xij
dvar boolean x[cities][cities];


// Auxiliar variable ui
dvar float+ u[cities];


// Objective function
minimize sum (i in cities) sum(j in cities) x[i][j] * Cost[i][j];


// Constraints
subject to {

    constraint_one:
    forall (i in cities) sum (j in cities: j!=i) x[i][j] == 1;

    constrain_two:
    forall (j in cities) sum (i in cities: i!=j) x[i][j] == 1;

    constraint_three:
    forall (i in cities : i >=2) forall(j in cities: j >=2 && j!=i) u[i] - u[j] + n * x[i][j] <= n - 1;
	
	starting_point:
    u[1] == 0;
}


// Print result
execute POSTPROCESS{
  	for (var i in cities)
        for (var j in cities)
            if (x[i][j] > 0)
            	writeln(i," ---> ",j, " cost: ", Cost[i][j]);
}