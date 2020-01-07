package networksimulator;

network NetworkNsfnet
{
    parameters:
        int numNodes = 14;
        int numTx = 14;

        @display("bgb=893,512");
    types:
        channel Channel extends ned.DatarateChannel
        {
            delay = 0ms;
            datarate = 10 kbps;
        }
    submodules:
        node0: Server {
            id = 0;
            numTx = numTx;
            numNodes = numNodes;
            @display("p=59,286");
            gates:
                port[3];
        }
        node1: Server {
            id = 1;
            numTx = numTx;
            numNodes = numNodes;
            @display("p=120,195");
            gates:
                port[3];
        }
        node2: Server {
            id = 2;
            numTx = numTx;
            numNodes = numNodes;
            @display("p=158,345");
            gates:
                port[3];
        }
        node3: Server {
            id = 3;
            numTx = numTx;
            numNodes = numNodes;
            @display("p=212,254");
            gates:
                port[3];
        }
        node4: Server {
            id = 4;
            numTx = numTx;
            numNodes = numNodes;
            @display("p=316,258");
            gates:
                port[3];
        }
        node5: Server {
            id = 5;
            numTx = numTx;
            numNodes = numNodes;
            @display("p=306,361");
            gates:
                port[4];
        }
        node6: Server {
            id = 6;
            numTx = numTx;
            numNodes = numNodes;
            @display("p=418,244");
            gates:
                port[2];
        }
        node7: Server {
            id = 7;
            numTx = numTx;
            numNodes = numNodes;
            @display("p=416,173");
            gates:
                port[3];
        }
        node8: Server {
            id = 8;
            numTx = numTx;
            numNodes = numNodes;
            @display("p=422,94");
            gates:
                port[3];
        }
        node9: Server {
            id = 9;
            numTx = numTx;
            numNodes = numNodes;
            @display("p=589,106");
            gates:
                port[3];
        }
        node10: Server {
            id = 10;
            numTx = numTx;
            numNodes = numNodes;
            @display("p=529,179");
            gates:
                port[4];
        }
        node11: Server {
            id = 11;
            numTx = numTx;
            numNodes = numNodes;
            @display("p=682,229");
            gates:
                port[3];
        }
        node12: Server {
            id = 12;
            numTx = numTx;
            numNodes = numNodes;
            @display("p=635,327");
            gates:
                port[3];
        }
        node13: Server {
            id = 13;
            numTx = numTx;
            numNodes = numNodes;
            @display("p=527,396");
            gates:
                port[2];
        }




        tController: NetTrafficController {
            numTx = numTx;
            numNodes = numNodes;
            @display("is=vs;p=460,24");
            gates:
                out[numTx];
        }

    connections:

        node0.port[0] <--> Channel <--> node1.port[0];
        node0.port[1] <--> Channel <--> node3.port[0];
        node0.port[2] <--> Channel <--> node2.port[0];
        node1.port[1] <--> Channel <--> node2.port[1];
        node1.port[2] <--> Channel <--> node7.port[0];
        node2.port[2] <--> Channel <--> node5.port[0];
        node3.port[1] <--> Channel <--> node4.port[0];
        node3.port[2] <--> Channel <--> node8.port[0];
        node4.port[1] <--> Channel <--> node5.port[1];
        node4.port[2] <--> Channel <--> node6.port[0];
        node5.port[2] <--> Channel <--> node12.port[0];
        node5.port[3] <--> Channel <--> node13.port[0];
        node6.port[1] <--> Channel <--> node7.port[1];
        node7.port[2] <--> Channel <--> node10.port[0];
        node8.port[1] <--> Channel <--> node9.port[0];
        node8.port[2] <--> Channel <--> node11.port[0];
        node9.port[1] <--> Channel <--> node10.port[1];
        node9.port[2] <--> Channel <--> node12.port[1];
        node10.port[2] <--> Channel <--> node11.port[1];
        node10.port[3] <--> Channel <--> node13.port[1];
        node11.port[2] <--> Channel <--> node12.port[2];




        tController.out[0] --> { @display("ls=grey,1,d"); } --> node0.tControl;
        tController.out[1] --> { @display("ls=grey,1,d"); } --> node1.tControl;
        tController.out[2] --> { @display("ls=grey,1,d"); } --> node2.tControl;
        tController.out[3] --> { @display("ls=grey,1,d"); } --> node3.tControl;
        tController.out[4] --> { @display("ls=grey,1,d"); } --> node4.tControl;
        tController.out[5] --> { @display("ls=grey,1,d"); } --> node5.tControl;
        tController.out[6] --> { @display("ls=grey,1,d"); } --> node6.tControl;
        tController.out[7] --> { @display("ls=grey,1,d"); } --> node7.tControl;
        tController.out[8] --> { @display("ls=grey,1,d"); } --> node8.tControl;
        tController.out[9] --> { @display("ls=grey,1,d"); } --> node9.tControl;
        tController.out[10] --> { @display("ls=grey,1,d"); } --> node10.tControl;
        tController.out[11] --> { @display("ls=grey,1,d"); } --> node11.tControl;
        tController.out[12] --> { @display("ls=grey,1,d"); } --> node12.tControl;
        tController.out[13] --> { @display("ls=grey,1,d"); } --> node13.tControl;

}
