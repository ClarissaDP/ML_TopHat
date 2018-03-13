#include "itkImage.h"
#include "itkGDCMImageIO.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkGrayscaleMorphologicalOpeningImageFilter.h"
#include "itkGrayscaleMorphologicalClosingImageFilter.h"
#include "itkBinaryBallStructuringElement.h"

#include "itkFlatStructuringElement.h"
#include "itkConstrainedValueAdditionImageFilter.h"
#include "itkConstrainedValueDifferenceImageFilter.h"

#include "itkStatisticsImageFilter.h"


// Exemplo de opening e closing com top-hat
//  -> https://itk.org/Doxygen/html/Examples_2Filtering_2MorphologicalImageEnhancement_8cxx-example.html#_a5


int main(int argc, char* argv[]) {

  //std::cerr << "Entrou: " << argv[1] << std::endl;

  if (argc < 2) {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << "DicomInputFileName" << std::endl;
    return EXIT_FAILURE;
  }
  
  unsigned int radius = 5;

  const char * inputFileName = argv[1];
  const char * outputFileName = "output.dcm";


  // Definir tipos
  typedef signed short    PixelType;
  const unsigned int      Dimension = 2;
  typedef itk::Image< PixelType, Dimension >  ImageType;
  typedef itk::ImageFileReader< ImageType >   ReaderType;


  // Ler imagem    
  ReaderType::Pointer reader = ReaderType::New();
  typedef itk::GDCMImageIO       ImageIOType;
  ImageIOType::Pointer dicomIO = ImageIOType::New();
  reader->SetFileName( inputFileName );
  reader->SetImageIO( dicomIO );

  try {
    reader->Update();
  }
  catch (itk::ExceptionObject &ex) {
    std::cout << ex << std::endl;
    return EXIT_FAILURE;
  }



  // elemento estruturante  
	//typedef itk::BinaryBallStructuringElement< PixelType, Dimension > StructuringElementType; 
	typedef itk::FlatStructuringElement< Dimension > StructuringElementType; 
	// define os tipos opening (abertura) e closing (fechamento)  
	typedef itk::GrayscaleMorphologicalOpeningImageFilter< ImageType, ImageType, StructuringElementType >  OpeningFilterType; 
	typedef itk::GrayscaleMorphologicalClosingImageFilter< ImageType, ImageType, StructuringElementType >  ClosingFilterType; 
	// define filtros de operacao aritmetica  
	typedef itk::ConstrainedValueAdditionImageFilter< ImageType, ImageType, ImageType > AdditionFilterType; 
	typedef itk::ConstrainedValueDifferenceImageFilter< ImageType, ImageType, ImageType > SubtractionFilterType;
  


	// Cria elemento estruturante 
  /*
  StructuringElementType  structuringElement;  
	structuringElement.SetRadius( radius );  
	structuringElement.CreateStructuringElement();
  */

	StructuringElementType::RadiusType elementRadius;
	elementRadius.Fill(radius);  
	StructuringElementType  structuringElement = StructuringElementType::Ball(elementRadius); 


	// Cria os filtros de opening closing   
	OpeningFilterType::Pointer  opening  = OpeningFilterType::New();
	ClosingFilterType::Pointer  closing  = ClosingFilterType::New();
	
  // Inicializa os metodos do opening e closing  
	opening->SetKernel(  structuringElement );
	closing->SetKernel(  structuringElement );
	
  // Seta imagem de entrada  
	opening->SetInput( reader->GetOutput() );
	closing->SetInput( reader->GetOutput() );
	
  // Imagem Original - Abertura 
	SubtractionFilterType::Pointer topHat = SubtractionFilterType::New();
	topHat->SetInput1( reader->GetOutput() );
	topHat->SetInput2( opening->GetOutput() );

  // Fechamento - Imagem Original  
	SubtractionFilterType::Pointer bottomHat = SubtractionFilterType::New();
	bottomHat->SetInput1( closing->GetOutput() );
	bottomHat->SetInput2( reader->GetOutput() );
	

  // Extrai estatisticas
  typedef itk::StatisticsImageFilter<ImageType> StatisticsImageFilterType;
  StatisticsImageFilterType::Pointer statisticsImageFilter = StatisticsImageFilterType::New();
  statisticsImageFilter->SetInput( topHat->GetOutput() );
  statisticsImageFilter->Update();
 
  std::cout << statisticsImageFilter->GetMean() << " ";
  std::cout << statisticsImageFilter->GetSigma() << " ";
  //std::cout << statisticsImageFilter->GetVariance() << " ";


  statisticsImageFilter->SetInput( bottomHat->GetOutput() );
  statisticsImageFilter->Update();
  
  std::cout << statisticsImageFilter->GetMean() << " ";
  std::cout << statisticsImageFilter->GetSigma() << " ";
  //std::cout << statisticsImageFilter->GetVariance() << " ";

    
  char c = inputFileName[0];
  int count = 0;
  for (int i = 0; c != '\n' && count < 3 ; ++i) {
    if ( count == 2 )
      std::cout << c ;
    if ( c == '/' )
      count++;
    c = inputFileName[i];
  }
  
  std::cout << " " << inputFileName[2] << std::endl;


  /*
  typedef itk::ImageFileWriter< ImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput( topHat->GetOutput() );
  writer->SetImageIO( dicomIO );
  writer->SetFileName( "topHat.dcm" );
  
  try {
    writer->Update();
  }
  catch( itk::ExceptionObject & error ) {
    std::cerr << "Error: " << error << std::endl;
    return EXIT_FAILURE;
  }

  
  writer->SetInput( bottomHat->GetOutput() );
  writer->SetImageIO( dicomIO );
  writer->SetFileName( "bottomHat.dcm" );
  
  try {
    writer->Update();
  }
  catch( itk::ExceptionObject & error ) {
    std::cerr << "Error: " << error << std::endl;
    return EXIT_FAILURE;
  }
  */


  return EXIT_SUCCESS;
}
