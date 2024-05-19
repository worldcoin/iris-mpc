use aws_sdk_sqs::{config::Region, Client, Error};
use clap::Parser;
use cudarc::driver::{
    result::{memcpy_dtoh_async, stream::synchronize},
    sys::lib,
    DevicePtr,
};
use gpu_iris_mpc::{
    setup::iris_db::{iris::IrisCodeArray, shamir_iris::ShamirIris},
    sqs::{SMPCRequest, SQSMessage},
};
use std::{env, fs::metadata, time::Instant};

use gpu_iris_mpc::{
    device_manager::DeviceManager,
    mmap::{read_mmap_file, write_mmap_file},
    preprocess_query,
    setup::{
        id::PartyID,
        iris_db::{db::IrisDB, iris::IrisCode, shamir_db::ShamirIrisDB},
        shamir::Shamir,
    },
    DistanceComparator, ShareDB,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

const REGION: &str = "us-east-2";
const DB_SIZE: usize = 8 * 1_000;
const QUERIES: usize = 930;
const RNG_SEED: u64 = 42;
const MAX_CONCURRENT_REQUESTS: usize = 5;
const DB_CODE_FILE: &str = "/opt/dlami/nvme/codes.db";
const DB_MASK_FILE: &str = "/opt/dlami/nvme/masks.db";

const SAMPLE_CODE: &[u64] = &[5128898750777317421u64, 10280194543220714962u64, 16283209011116730021u64, 16487406209377363082u64, 1042313098880278384u64, 785357641948052250u64, 13135470951968037u64, 15580461643097350336u64, 2988219270918274421u64, 15680032030599556387u64, 15052219194448633467u64, 17738982131066720167u64, 7324632261415590942u64, 822688419325357736u64, 9951672840605536170u64, 928620735742417156u64, 8892958599908899115u64, 86801274824384315u64, 9895622637843603040u64, 16257878912899846414u64, 3238528739466500854u64, 11257340220756936282u64, 16785433421118559302u64, 10995566987393511037u64, 14725984146529030054u64, 11459617555587003780u64, 13839657046409468567u64, 16712305675756561510u64, 14266721495970095659u64, 11923006250513659293u64, 16068950041054207704u64, 8334163256367006417u64, 585390396684721778u64, 6516224943617132632u64, 3746052973320446943u64, 1205937242960317128u64, 983023515747785059u64, 12292808740701135374u64, 7599170843606724078u64, 2104048582286095245u64, 854668057396022900u64, 9194133030784588259u64, 17664933446844065898u64, 16891515866243786611u64, 13793552823760664882u64, 3033633897461106533u64, 6586578053723610145u64, 16315825958698182158u64, 9971120029616414964u64, 7359993378792253914u64, 1889973034227702138u64, 17804914884400851322u64, 2001725066290077834u64, 1436492728640398655u64, 10094425026549017543u64, 2984564153915292597u64, 4767860825737519115u64, 7340246266592855053u64, 8302067216026857347u64, 7157696224919831784u64, 17779352572237167171u64, 2094168268416861584u64, 17831144056407254625u64, 6799375502332345457u64, 11464990271495852310u64, 4045450974074834083u64, 17714479885699984458u64, 15473681579140272908u64, 18143707359750615394u64, 15850022182176129252u64, 14648756053226077927u64, 6814596081435907606u64, 12314346887856359517u64, 3435549532838257074u64, 5539264006477101549u64, 9507547829956974110u64, 3055644552379925607u64, 10163716221890150346u64, 15177563974974995571u64, 16888304954161489028u64, 3828481034062783108u64, 17695991502848523370u64, 4522064485586147935u64, 15555711433672829253u64, 5310893819803910550u64, 1418309228494828597u64, 17789225019948856658u64, 5563686357511723839u64, 4540689236601976436u64, 670348583075773460u64, 11411866972290101251u64, 8725755005552357334u64, 13018756262971946336u64, 17420106905376633566u64, 8361167932988803658u64, 430525176894582856u64, 910122439175936147u64, 5581800698698233787u64, 17710195502459160721u64, 6518542591638258696u64, 10151302921132619185u64, 11871349792387302405u64, 304803775664233037u64, 7290929183869219083u64, 9449695530491019480u64, 4445961663614349977u64, 5106119364206713771u64, 2598089708233774266u64, 1786067004536850336u64, 12989377731415499520u64, 17154198001696586018u64, 16911796363614924767u64, 967823388456067277u64, 15128084739206413973u64, 17720887780443953527u64, 433002393933469669u64, 14334912199714652862u64, 10223596862441141632u64, 6541736607209104273u64, 6365371100174507714u64, 9365432573519593527u64, 3768635713756607744u64, 8417517202185344183u64, 7959450697174380366u64, 11683419947409628964u64, 9507671235913326904u64, 10987401963546459750u64, 11165916223074915952u64, 521357834498322160u64, 7102096299690570673u64, 15260857583033687625u64, 3561252935509274566u64, 15294899733804802033u64, 481346315807809798u64, 3763854928429517203u64, 3324775196060179818u64, 6438953771120785676u64, 13840094077420844020u64, 5607864461449461986u64, 7698695196576251366u64, 3948495525723547434u64, 5013391251354925014u64, 17238999009379963968u64, 16735522640066795759u64, 9514232026655996227u64, 6279516196633834748u64, 1624401151277145560u64, 4679342005609525920u64, 11346829233641335789u64, 3276458614464833882u64, 7243931965173494254u64, 15598263500932228358u64, 8774135507152511440u64, 13750400387576216762u64, 10632147853191339100u64, 6028109513858363414u64, 8561785496340049643u64, 8579591373148155761u64, 10468510225480917160u64, 47936750718859951u64, 15872968296042290430u64, 1949250508431500295u64, 8082652673574245400u64, 9399148735725718447u64, 1094694583481783192u64, 16533583874073102888u64, 14681149906274712740u64, 7745457468073852552u64, 17991924321504192950u64, 3389989604489140620u64, 7837428923758487962u64, 11192630472473328171u64, 3882953140123657460u64, 3070558673822053065u64, 3606635674368986421u64, 3080280792361974854u64, 9966968727888705971u64, 6735586448205424417u64, 14044590846480619733u64, 4319454957759237413u64, 8175496350624801015u64, 8729898397975868743u64, 6286242064088361632u64, 17030538670942809466u64, 6549367334102784853u64, 7811874819944175951u64, 14533149521225066445u64, 7344384424710044451u64, 6400546778879023675u64, 17647539757289959937u64, 1464814881070055351u64, 2921901899283338495u64, 16473728216830342289u64, 13611612213464022810u64, 36543380413985920u64, 5005795199634306235u64, 13524255070852342709u64, 9841717850462182801u64, 9616160235043479110u64, 6346705145125838096u64];
const SAMPLE_MASK: &[u64] = &[9209861235791626239u64, 12970295939607560143u64, 8569205774159421438u64, 18148361562555809783u64, 9218848636354289151u64, 18428025975946543095u64, 18157949097187737471u64, 18446735208728229119u64, 16140901054765071295u64, 16140267728618318847u64, 18153869085546381311u64, 18445055212038111231u64, 17221764975062417067u64, 18444491173813809143u64, 17834243529203505855u64, 18446741857506423287u64, 18444314267763605495u64, 9203105820008972287u64, 9187044155484536319u64, 9223368735834767207u64, 18442170070978265023u64, 18446673704964325375u64, 13258596056280563709u64, 18442240469248243583u64, 18158370761044852703u64, 18302628885091581943u64, 18446717681068277755u64, 18446724263172898751u64, 18446724281924059007u64, 18437736874450598909u64, 17813988291704221695u64, 18446744073675997055u64, 9223371759812575231u64, 18428729603783645183u64, 18446741315094313470u64, 18156261893449187327u64, 18446744065052508103u64, 9221080654605713279u64, 18302628816880631807u64, 18428720874644307454u64, 18444492273744871287u64, 15798625293758889983u64, 18446744071562067966u64, 18446738575077670271u64, 18446744073709535231u64, 16140338114542423807u64, 18446732803445882619u64, 18374684280648351741u64, 18446744073709551610u64, 11529206249637756919u64, 16140901064493752319u64, 18446726479376003071u64, 17203749477010110463u64, 18302610709063520253u64, 18373560579764748219u64, 9223370928719396830u64, 13794525656217812991u64, 18401708003211337727u64, 13253811506753040383u64, 18444492239536082938u64, 18446744004922965759u64, 4611676658083037183u64, 18446744056529551359u64, 18446741874686032895u64, 18445475237291090943u64, 17275807886817089523u64, 17291569669775129599u64, 18445618173802643455u64, 18435485053568941931u64, 18446743523684253567u64, 9223371474214060029u64, 18446726438573832187u64, 17212524678537478143u64, 18122480227480231871u64, 9223354306692906814u64, 18441114573898448637u64, 8644586367180865519u64, 13835058055281901567u64, 17852268922896646135u64, 17867468434164416511u64, 18446742972016877567u64, 18446726446089756607u64, 18158513628704145407u64, 18410715272395489275u64, 18374545192426925819u64, 17293679632580591615u64, 18409519007971147771u64, 18446744038778257405u64, 18302610743524064767u64, 18446735140176527359u64, 18410591993949257695u64, 15546421446782213119u64, 17847202364725886699u64, 18446708888258985983u64, 18446744071562054655u64, 17293822569102700543u64, 18433223241783830399u64, 18365114131440197631u64, 17865216771825073791u64, 18445618171118354415u64, 9223349762885943291u64, 18437664297829006838u64, 18428624122083803134u64, 18446744073139126271u64, 18446708871754938879u64, 18428588936637708223u64, 18446739400785002493u64, 17217261237963833311u64, 18158513521463914367u64, 18410715276622393341u64, 18446708889113837439u64, 18338587309613350653u64, 18446743996391686139u64, 18013813569261795831u64, 17870282702928732031u64, 9149901294406713215u64, 18446603335681013759u64, 18437725570098782175u64, 18338582915809542143u64, 18301784426343811063u64, 18428729675132959741u64, 18369056980137409535u64, 8935141658555580415u64, 16140813103431286781u64, 18446744065039794175u64, 18427603775292947455u64, 18439988390263783423u64, 18446744065119617023u64, 18442239373427605503u64, 18446743935062638591u64, 18014398372043025403u64, 9187316778539151347u64, 13256341069609565939u64, 18302624487587045375u64, 18302628885633687543u64, 17942340915441695738u64, 18446726481523439615u64, 18444421853607866366u64, 18427601026245722111u64, 16057001721368395263u64, 18410715207904002047u64, 17870283320192925695u64, 16094739099595807487u64, 9218305450765713343u64, 18446743485298462719u64, 18428720861924947711u64, 18086456103519911935u64, 13834985483084987903u64, 18158443328813662191u64, 18446603331657760765u64, 18428725277019332606u64, 18410714725784483839u64, 18446672588223545311u64, 17293804976747315135u64, 17221759475359079935u64, 18446735277611216895u64, 18446735277616529407u64, 18446706690311733149u64, 17293822259865058297u64, 18374682072497258495u64, 18446742321329340399u64, 13528778096232103918u64, 13546827679122063359u64, 18446744056251752431u64, 18374123529718193598u64, 18408461204839202814u64, 17865779719625900031u64, 16140335881159438198u64, 18446656112777199535u64, 18445618102935732223u64, 18374686473229106687u64, 18446744004987976695u64, 18158512872923856382u64, 18446742969768706039u64, 17870283287012823935u64, 18446744073673834495u64, 17293822019313336239u64, 18446585744033838527u64, 13835055855185166335u64, 18428447650180235263u64, 18446603336217001979u64, 18014389155043211229u64, 8065946932482142175u64, 9214081056626360127u64, 17140664996590518271u64, 14411518807585570807u64, 18446176725302247351u64, 13834495098886258687u64, 17870283317094371327u64, 17870283312715398943u64, 18441113474661449471u64, 18446673696356560383u64, 18230289816610664443u64, 18145002890085793781u64, 13835020671877365755u64, 18374651295156895487u64, 17275667422367449055u64, 18446744073703260095u64, 18428729666610126847u64, 18437736873917833215u64];


#[derive(Debug, Parser)]
struct Opt {
    #[structopt(short, long)]
    queue: String,

    #[structopt(short, long)]
    party_id: usize,

    #[structopt(short, long)]
    bootstrap_url: Option<String>,
}

async fn receive_batch(client: &Client, queue_url: &String, party_id: usize) -> eyre::Result<Vec<ShamirIris>> {
    let mut batch = vec![];

    let mut tmp_code: [u64; 200] = [0u64; 200];
    let mut tmp_mask: [u64; 200] = [0u64; 200];

    tmp_code.copy_from_slice(SAMPLE_CODE);
    tmp_mask.copy_from_slice(SAMPLE_MASK);

    let mut rng = StdRng::seed_from_u64(1337);
    let template = IrisCode { code: IrisCodeArray(tmp_code), mask: IrisCodeArray(tmp_mask) };
    let iris = ShamirIris::share_iris(&template, &mut rng);

    while batch.len() < QUERIES {
        batch.push(iris[party_id].clone());
        // let rcv_message_output = client
        //     .receive_message()
        //     .max_number_of_messages(1i32)
        //     .queue_url(queue_url)
        //     .send()
        //     .await?;

        // for sns_message in rcv_message_output.messages.unwrap_or_default() {
        //     let message: SQSMessage = serde_json::from_str(sns_message.body().unwrap())?;
        //     let message: SMPCRequest = serde_json::from_str(&message.message)?;

        //     let iris: ShamirIris = message.into();

        //     batch.extend(iris.all_rotations());
        //     // TODO: we should only delete after processing
        //     client
        //         .delete_message()
        //         .queue_url(queue_url)
        //         .receipt_handle(sns_message.receipt_handle.unwrap())
        //         .send()
        //         .await?;
        // }
    }

    Ok(batch)
}

fn prepare_query_batch(batch: Vec<ShamirIris>) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let (code_queries, mask_queries): (Vec<Vec<u16>>, Vec<Vec<u16>>) = batch
        .iter()
        .map(|iris| (iris.code.to_vec(), iris.mask.to_vec()))
        .unzip();

    let code_query = preprocess_query(&code_queries.into_iter().flatten().collect::<Vec<_>>());
    let mask_query = preprocess_query(&mask_queries.into_iter().flatten().collect::<Vec<_>>());
    (code_query, mask_query)
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let Opt {
        queue,
        party_id,
        bootstrap_url,
    } = Opt::parse();

    let region_provider = Region::new(REGION);
    let shared_config = aws_config::from_env().region(region_provider).load().await;
    let client = Client::new(&shared_config);

    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let seed0 = rng.gen::<[u32; 8]>();
    let seed1 = rng.gen::<[u32; 8]>();
    let seed2 = rng.gen::<[u32; 8]>();

    // Init RNGs
    let chacha_seeds = match party_id {
        0 => (seed0, seed2),
        1 => (seed1, seed0),
        2 => (seed2, seed1),
        _ => unimplemented!(),
    };

    let l_coeff = Shamir::my_lagrange_coeff_d2(PartyID::try_from(party_id as u8).unwrap());

    // Generate or load DB
    let (codes_db, masks_db) = if metadata(DB_CODE_FILE).is_ok() && metadata(DB_MASK_FILE).is_ok() {
        (read_mmap_file(DB_CODE_FILE)?, read_mmap_file(DB_CODE_FILE)?)
    } else {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let db = IrisDB::new_random_par(DB_SIZE, &mut rng);

        println!("{:?}", db.db[1]);

        let shamir_db = ShamirIrisDB::share_db_par(&db, &mut rng);

        let codes_db = shamir_db[party_id]
            .db
            .iter()
            .flat_map(|entry| entry.code)
            .collect::<Vec<_>>();

        let masks_db = shamir_db[party_id]
            .db
            .iter()
            .flat_map(|entry| entry.mask)
            .collect::<Vec<_>>();

        write_mmap_file(DB_CODE_FILE, &codes_db)?;
        write_mmap_file(DB_MASK_FILE, &masks_db)?;
        (codes_db, masks_db)
    };

    println!("Starting engines...");

    let device_manager = DeviceManager::init();

    let mut codes_engine = ShareDB::init(
        party_id,
        device_manager.clone(),
        l_coeff,
        &codes_db,
        QUERIES,
        chacha_seeds,
        bootstrap_url.clone(),
        Some(true),
        Some(3000),
    );

    println!("Codes Engines ready!");

    let mut masks_engine = ShareDB::init(
        party_id,
        device_manager.clone(),
        l_coeff,
        &masks_db,
        QUERIES,
        chacha_seeds,
        bootstrap_url.clone(),
        Some(true),
        Some(3001),
    );
    let mut distance_comparator = DistanceComparator::init(DB_SIZE, QUERIES);

    println!("Engines ready!");

    // Prepare streams etc.
    let mut streams = vec![];
    let mut cublas_handles = vec![];
    let mut results = vec![];
    for _ in 0..MAX_CONCURRENT_REQUESTS {
        let tmp_streams = device_manager.fork_streams();
        cublas_handles.push(device_manager.create_cublas(&tmp_streams));
        streams.push(tmp_streams);
        results.push(distance_comparator.prepare_results());
    }

    // Main Loop
    let mut current_dot_event = device_manager.create_events();
    let mut next_dot_event = device_manager.create_events();
    let mut current_exchange_event = device_manager.create_events();
    let mut next_exchange_event = device_manager.create_events();
    let mut request_counter = 0;

    loop {
        let batch = receive_batch(&client, &queue, party_id).await?;
        let (code_query, mask_query) = prepare_query_batch(batch);

        println!("Received new batch.");

        let request_streams = &streams[request_counter % MAX_CONCURRENT_REQUESTS];
        let request_cublas_handles = &cublas_handles[request_counter % MAX_CONCURRENT_REQUESTS];

        // First stream doesn't need to wait on anyone
        if request_counter == 0 {
            device_manager.record_event(request_streams, &current_dot_event);
            device_manager.record_event(request_streams, &current_exchange_event);
        }

        // BLOCK 1: calculate individual dot products
        device_manager.await_event(request_streams, &current_dot_event);
        codes_engine.dot(&code_query, request_streams, request_cublas_handles);
        masks_engine.dot(&mask_query, request_streams, request_cublas_handles);

        // BLOCK 2: calculate final dot product result, exchange and compare
        device_manager.await_event(request_streams, &current_exchange_event);
        codes_engine.dot_reduce(request_streams);
        masks_engine.dot_reduce(request_streams);

        device_manager.record_event(request_streams, &next_dot_event);

        codes_engine.exchange_results(request_streams);
        masks_engine.exchange_results(request_streams);
        distance_comparator.reconstruct_and_compare(
            &codes_engine.results_peers,
            &masks_engine.results_peers,
            request_streams,
            results[request_counter % MAX_CONCURRENT_REQUESTS]
                .iter()
                .map(|r| *r.device_ptr())
                .collect::<Vec<_>>(),
        );

        device_manager.record_event(request_streams, &next_exchange_event);

        // Start thread to wait for the results
        let tmp_streams = streams[request_counter % MAX_CONCURRENT_REQUESTS]
            .iter()
            .map(|s| s.stream as u64)
            .collect::<Vec<_>>();
        let tmp_devs = distance_comparator.devs.clone();
        let tmp_results: Vec<u64> = results[request_counter % MAX_CONCURRENT_REQUESTS]
            .iter()
            .map(|r| *r.device_ptr())
            .collect::<Vec<_>>();

        device_manager.await_streams(&request_streams);

        // tokio::spawn(async move {
            let mut index_results = vec![];
            for i in 0..tmp_devs.len() {
                tmp_devs[i].bind_to_thread().unwrap();
                let mut tmp_result = vec![0u32; QUERIES];
                unsafe {
                    lib()
                        .cuMemcpyDtoHAsync_v2(
                            tmp_result.as_mut_ptr() as *mut _,
                            tmp_results[i],
                            QUERIES * std::mem::size_of::<u32>(),
                            tmp_streams[i] as *mut _,
                        )
                        .result()
                        .unwrap();
                    synchronize(tmp_streams[i] as *mut _).unwrap();
                }
                index_results.push(tmp_result);
            }

            for j in 0..8 {
                print!("{:?} ", index_results[j][0]);
            }
            println!("");
        // });

        // Prepare for next batch
        request_counter += 1;
        current_dot_event = next_dot_event;
        current_exchange_event = next_exchange_event;
        next_dot_event = device_manager.create_events();
        next_exchange_event = device_manager.create_events();
    }

    Ok(())
}
