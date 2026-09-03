/*
  Feeding experimentation device 3 (FED3)
  Classic FED3 script
  This script mimics the classic FED3 menuing system for selecting among the following programs

  // FEDmodes:
  // 0 Free feeding
  // 1 FR1
  // 2 FR3
  // 3 FR5
  // 4 Progressive Ratio
  // 5 Extinction
  // 6 Light tracking FR1 task
  // 7 FR1 (reversed)
  // 8 PR (reversed)
  // 9 Optogenetic stimulation
  // 10 Optogenetic stimulation (reversed)
  // 11 Timed FR1 feeding

  alexxai@wustl.edu
  December, 2020

  This project is released under the terms of the Creative Commons - Attribution - ShareAlike 3.0 license:
  human readable: https://creativecommons.org/licenses/by-sa/3.0/
  legal wording: https://creativecommons.org/licenses/by-sa/3.0/legalcode
  Copyright (c) 2020 Lex Kravitz

*/

#include <FED3.h>                //Include the FED3 library 
String sketch = "Classic";       //Unique identifier text for each sketch
FED3 fed3 (sketch);              //Start the FED3 object

//variables for PR tasks
int poke_num = 0;                                      // this variable counts active pokes in the PR loop since the last reset
int pokes_required = 1;                                // set the initial PR requirement to 1 poke

void setup() {
  fed3.ClassicFED3 = true;
  fed3.begin();                                        //Setup the FED3 hardware
}

void loop() {
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                                                     Mode 0: Free feeding
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  if (fed3.FEDmode == 0) {
    fed3.sessiontype = "Free_feed";                     //The text in "sessiontype" will appear on the screen and in the logfile
    fed3.DisplayPokes = false;                          //Turn off poke indicators for free feeding mode
    fed3.UpdateDisplay();                               //Update display for free feeding session to remove poke display (they are on by default)
    fed3.Feed();
    fed3.BNC(50, 1);                                    //Deliver 1 pulse at 10Hz (50ms HIGH, 50ms LOW), lasting 100ms
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                                                     Modes 1-3: Fixed Ratio Programs FR1, FR3, FR5
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  if ((fed3.FEDmode == 1) or (fed3.FEDmode == 2) or (fed3.FEDmode == 3)) {
    if (fed3.FEDmode == 1) fed3.sessiontype = "FR1";    //The text in "sessiontype" will appear on the screen and in the logfile
    if (fed3.FEDmode == 2) fed3.sessiontype = "FR3";    //The text in "sessiontype" will appear on the screen and in the logfile
    if (fed3.FEDmode == 3) fed3.sessiontype = "FR5";    //The text in "sessiontype" will appear on the screen and in the logfile
    if (fed3.Left) {
      fed3.logLeftPoke();                               //Log left poke
      fed3.BNC(45, 1);                                //Deliver 1 pulse at approximately 11.1Hz (45ms HIGH, 45ms LOW), lasting 90ms
      if (fed3.LeftCount % fed3.FR == 0) {              //if fixed ratio is  met
        fed3.BNC(45, 1);                                //Deliver 1 additional pulse at approximately 11.1Hz (45ms HIGH, 45ms LOW), lasting 90ms
        fed3.ConditionedStimulus();                     //deliver conditioned stimulus (tone and lights)
        fed3.Feed();                                    //deliver pellet
        fed3.BNC(45, 3);                                //Deliver 3 pulses at approximately 11.1Hz (45ms HIGH, 45ms LOW), lasting 270ms
      }
    }
    if (fed3.Right) {                                    //If right poke is triggered
      fed3.logRightPoke();
      fed3.BNC(30, 5);  
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                                                     Mode 4: Progressive Ratio
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  if (fed3.FEDmode == 4) {
    fed3.sessiontype = "ProgRatio";                      //The text in "sessiontype" will appear on the screen and in the logfile
    if (fed3.Left) {                                     //If left poke is triggered
      fed3.logLeftPoke();                                //Log left poke
      fed3.BNC(45, 1);                                 //Deliver 1 pulse at approximately 11.1Hz (45ms HIGH, 45ms LOW), lasting 90ms
      fed3.Click();                                      //Click
      poke_num++;                                        //increment the active poke count for this PR trial
      if (poke_num == pokes_required) {                  //check to see if the mouse has achieved the correct number of pokes in order to receive the pellet
        fed3.ConditionedStimulus();                      //Deliver conditioned stimulus (tone and lights)  
        fed3.BNC(45, 1);                                 //Deliver 1 additional pulse at approximately 11.1Hz (45ms HIGH, 45ms LOW), lasting 90ms
        fed3.Feed();                                     //Deliver pellet
        fed3.BNC(45, 3);                                 //Deliver 3 pulses at approximately 11.1Hz (45ms HIGH, 45ms LOW), lasting 270ms

        // *** UPDATED PR FORMULA (MATCHES CLASSIC FED3 CODE) ***
        pokes_required = round((5 * exp((fed3.PelletCount + 1) * 0.2)) - 5);  // Progressive ratio exponential formula
        fed3.FR = pokes_required;                                             // Update FR requirement

        poke_num = 0;                                    //reset poke_num back to 0 for the next trial
      }
    }
    if (fed3.Right) {                                    //If right poke is triggered
      fed3.logRightPoke();
      fed3.BNC(30, 5);                                 //Deliver 5 pulses at approximately 16.7Hz (30ms HIGH, 30ms LOW), lasting 300ms
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                                                     Mode 5: Extinction
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  if (fed3.FEDmode == 5) {
    fed3.sessiontype = "Extinct";                        //The text in "sessiontype" will appear on the screen and in the logfile
    if (fed3.Left) {
      fed3.logLeftPoke();                                //Log left poke
      fed3.BNC(50, 1);   
      fed3.ConditionedStimulus();                        //deliver conditioned stimulus (tone and lights)
    }

    if (fed3.Right) {                                    //If right poke is triggered
      fed3.logRightPoke();
      fed3.BNC(30, 5);
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                                                    Mode 6: Light tracking FR1 task
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  if (fed3.FEDmode == 6) {
    fed3.sessiontype = "Light Trk";                       //The text in "sessiontype" will appear on the screen and in the logfile
    fed3.disableSleep();                                  //Sleep mode shuts the NeoPixels off to save power.  Therefore to leave pixels on during this task we must disable sleep mode.

    //If left poke is active, run FR1 session with left active
    if (fed3.activePoke == 1) {

      //Comment one of these next two lines out, depending on if you have FED3 or FED3.1 (with nosepoke lights)
            fed3.leftPokePixel(5,5,5,0) ;                       //turn on pixel inside left nosepoke dim white
//      fed3.leftPixel(5, 5, 5, 5) ;                       //turn on left-most pixel on Neopixel strip

      if (fed3.Left) {
        fed3.logLeftPoke();                               //Log left poke
        fed3.ConditionedStimulus();                       //deliver conditioned stimulus (tone and lights)
        fed3.Feed();
        fed3.randomizeActivePoke(3);                      //randomize which poke is active, specifying maximum on the same poke before forcing a switch
      }
      if (fed3.Right) {                                   //If right poke is triggered
        fed3.logRightPoke();
      }
    }
    //If right poke is active, run FR1 session with right active
    if (fed3.activePoke == 0) {

      //Comment one of these next two lines out, depending on if you have FED3 or FED3.1 (with nosepoke lights)
            fed3.rightPokePixel(5,5,5,0) ;                    //turn on pixel inside right nosepoke dim white
//      fed3.rightPixel(5, 5, 5, 5) ;                      //turn on right-most pixel on Neopixel strip

      if (fed3.Right) {
        fed3.logRightPoke();                              //Log right poke
        fed3.ConditionedStimulus();                       //deliver conditioned stimulus (tone and lights)
        fed3.Feed();                                      //deliver pellet
        fed3.randomizeActivePoke(3);                      //randomize which poke is active, specifying maximum on the same poke before forcing a switch
      }
      if (fed3.Left) {                                    //If left poke is triggered
        fed3.logLeftPoke();
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                                                     Mode 7: FR1 (reversed)
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  if (fed3.FEDmode == 7) {
    fed3.sessiontype = "FR1_R";                           //The text in "sessiontype" will appear on the screen and in the logfile
    fed3.activePoke = 0;                                  //Set activePoke to 0 to make right poke active
    if (fed3.Left) {                                      //If left poke
      fed3.logLeftPoke();                                 //Log left poke
      fed3.BNC(30, 5);  
    }
    if (fed3.Right) {                                     //If right poke is triggered
      fed3.logRightPoke();                                //Log Right Poke
       fed3.BNC(50, 1);  
      fed3.ConditionedStimulus();                         //Deliver conditioned stimulus (tone and lights)
      fed3.Feed();                                        //deliver pellet
       fed3.BNC(50, 3);  
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                                                     Mode 8: PR (reversed)
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  if (fed3.FEDmode == 8) {
    fed3.sessiontype = "PR_R";                          //The text in "sessiontype" will appear on the screen and in the logfile
    fed3.activePoke = 0;                                //Right poke is active
    if (fed3.Right) {                                   //If Right poke is triggered
      fed3.logRightPoke();                              //Log Right poke
      fed3.BNC(50, 3);   
      poke_num++;                                       //increment the active poke count for this PR trial
      if (poke_num == pokes_required) {                 //check to see if the mouse has achieved the correct number of pokes in order to receive the pellet
        fed3.ConditionedStimulus();                     //Deliver conditioned stimulus (tone and lights)
        fed3.Feed();                                    //Deliver pellet
        fed3.BNC(50, 1);   

        // *** UPDATED PR FORMULA (MATCHES CLASSIC FED3 CODE) ***
        pokes_required = round((5 * exp((fed3.PelletCount + 1) * 0.2)) - 5);  // Progressive ratio exponential formula  
        fed3.FR = pokes_required;                                             // Update FR requirement

        poke_num = 0;                                   //reset the number of pokes back to 0, for the next trial
        fed3.Right = false;
      }
      else {
        fed3.Click();                                   //If not enough pokes, just do a Click
      }
    }
    if (fed3.Left) {                                    //If left poke is triggered
      fed3.logLeftPoke();
      fed3.BNC(50, 2);   
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                                                     Mode 9: Optogenetic stimulation
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  if (fed3.FEDmode == 9) {
    fed3.sessiontype = "OptoStim";                      //The text in "sessiontype" will appear on the screen and in the logfile
    if (fed3.Left) {                                    //If left poke
      fed3.logLeftPoke();                               //Log left poke
      fed3.ConditionedStimulus();                       //Deliver conditioned stimulus (tone and lights)
      fed3.BNC(25, 20);                                 //Deliver 20 pulses at 20Hz (25ms HIGH, 25ms LOW), lasting 1 second
    }
    if (fed3.Right) {                                   //If right poke is triggered
      fed3.logRightPoke();
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                                                     Mode 10: Optogenetic stimulation (reversed)
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  if (fed3.FEDmode == 10) {
    fed3.sessiontype = "OptoStim_R";                     //The text in "sessiontype" will appear on the screen and in the logfile
    fed3.activePoke = 0;                                 //Set activePoke to 0 to make right poke active
    if (fed3.Right) {                                    //If Right poke
      fed3.logRightPoke();                               //Log Right poke
      fed3.ConditionedStimulus();                        //Deliver conditioned stimulus (tone and lights)
      fed3.BNC(25, 20);                                  //Deliver 20 pulses at 20Hz (25ms HIGH, 25ms LOW), lasting 1 second
    }
    if (fed3.Left) {                                     //If Left poke is triggered
      fed3.logLeftPoke();                                //Log LeftPoke
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                                                     Mode 11: Timed FR1 Feeding
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  if (fed3.FEDmode == 11) {
    fed3.sessiontype = "Timed";                         //The text in "sessiontype" will appear on the screen and in the logfile
    fed3.DisplayPokes = true;                          //Turn on poke indicators for timed FR1 feeding mode
    fed3.DisplayTimed = false;                           //Turn off timed feeding info
    fed3.UpdateDisplay();
    if (fed3.Left) {
    if (fed3.currentHour >= fed3.timedStart && fed3.currentHour < fed3.timedEnd) {     //If left poke is triggered and it's between the specified times
      fed3.logLeftPoke();                                                              //Log left poke
      fed3.ConditionedStimulus();                                                      //Deliver conditioned stimulus (tone and lights for 200ms)
      fed3.Feed();                                                                     //Deliver pellet
    }
    else {                                                                             //If it's not between the specified times
      fed3.logLeftPoke();                                                              //Log left poke
    }
  }

  if (fed3.Right) {                                                                    //If right poke is triggered
    fed3.logRightPoke();                                                               //Log right poke
  }
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                                                    Call fed3.run at least once per loop
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  fed3.run();
}
